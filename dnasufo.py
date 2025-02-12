from pathlib import Path
import numpy as np
import scipy.ndimage as ndi
from skimage import segmentation
from skimage import measure
from skimage.registration import optical_flow_ilk
import tifffile
import pandas as pd
from cellpose import models
import h5py
import matplotlib.pyplot as plt
import colorsys
import math
import trackpy as tp
import io


def contrast(x: np.ndarray):
    """Stretch the contrast to 0,1"""
    return (x - x.min()) / (x.max() - x.min())


def uv2rgb(x: np.ndarray):
    """Transform a two channel image to rgb"""
    return np.stack([contrast(x[0]), contrast(x[1]), contrast(x[0])], -1)


def segment_watershed(mask: np.ndarray, r: float = 5.0):
    d = ndi.gaussian_filter(ndi.distance_transform_edt(mask), r / 2.0) * mask
    seed = (ndi.label(d == ndi.maximum_filter(d, r))[0] * mask).astype(np.uint32)
    labels = segmentation.watershed(-d, seed) * mask
    labels = np.unique(labels, return_inverse=1)[1].reshape(labels.shape)
    return labels.astype(np.uint32)


def preprocess(img: np.ndarray, background: float, scale=[1, 0, 0.85, 0.85], niter=10):
    """Preprocess the image sequence with a gaussian blur"""

    u = np.maximum(
        ndi.gaussian_filter(img.astype(float) - background, scale),
        0,
    )

    for _ in range(niter):
        u = u * ndi.gaussian_filter(
            img / ndi.gaussian_filter(u + background, [0, 0, 2, 2]),
            [0, 0, 2, 2],
        )
    return u

    # return np.maximum(
    #     ndi.median_filter(
    #         ndi.gaussian_filter(img.astype(float) - background, [3, 0, 1, 1]),
    #         [3, 1, 1, 1],
    #     ),
    #     0,
    # )


def link(df, center):
    """Link one objects over time using the nearest neighbor"""
    trj = []
    for frame in range(df["frame"].max()):
        sdf = df[df["frame"] == frame]
        if len(sdf) == 1:
            trj.append(sdf.iloc[0])
        elif len(sdf) > 1:
            p = np.stack((sdf["centroid-0"], sdf["centroid-1"]), axis=1)
            k = np.argmin(np.linalg.norm(p - center, axis=1))
            if frame > 1:
                center = (center + np.array(p[k])) / 2
            else:
                center = np.array(p[k])
            trj.append(sdf.iloc[k])
    trj = pd.DataFrame(trj)
    trj["particle"] = 1
    return trj


def segment_and_track(img: np.ndarray):
    """Segment and track centermost objects

    Parameter
    ---------
    img : np.narray
        input with shape [T,C,H,W]

    Returns
    -------
    mask: np.ndarray
        all masks [T,H,W]
    position: np.ndarray
        [T,2] position
    """
    model = models.Cellpose(True, "cyto")

    center = np.array([img.shape[2] / 2, img.shape[3] / 2])
    mask = []
    position = []

    for t, frame in enumerate(img):
        labels = model.eval(frame, diameter=22)[0]

        if labels.max() == 0:
            labels = segment_watershed((frame[0] + frame[1]) > 5)

        if labels.max() == 0:
            raise Exception(f"No cell found at frame {t}")

        indices = np.unique(labels)[1:]
        p = ndi.center_of_mass(labels, labels, indices)
        k = np.argmin(np.linalg.norm(p - center, axis=1))
        mask.append(labels == indices[k])
        if t > 2:
            center = 0.5 * (np.array(p[k]) + center)
        else:
            center = np.array(p[k])
        position.append(center)

    mask = np.stack(mask)
    position = np.stack(position)
    return np.expand_dims(mask, 1), position


def segment_and_track_cell(img: np.ndarray, model=None):
    """Segment and track centermost objects

    Parameter
    ---------
    img : np.narray
        input with shape [T,C,H,W]
    model: Cellpose
        Cellpose model

    Returns
    -------
    mask: np.ndarray
        all masks [T,H,W]
    position: np.ndarray
        [T,2] position
    """
    if model is None:
        model = models.Cellpose(True, "cyto2")

    shp = img.shape[0], 1, img.shape[2], img.shape[3]
    labels = np.zeros(shp, dtype=np.uint32)
    df = []
    miss_counter = 0
    for t, frame in enumerate(img):
        # segment the cells
        method = "cellpose"
        labels[t, 0] = model.eval(frame, diameter=22)[0]

        if labels[t, 0].max() == 0:
            method = "watershed"
            labels[t, 0] = segment_watershed((frame[0] + frame[1]) > 5)

        if labels[t, 0].max() == 0:
            method = "watershed"
            labels[t, 0] = segment_watershed((frame[0] + frame[1]) > 4)

        if labels[t, 0].max() == 0:
            if t > 0:
                method = "propagate"
                labels[t, 0] = labels[t - 1, 0]
                miss_counter = miss_counter + 1
            else:
                raise Exception("No cell found on first frame.")

        if miss_counter > 5:
            print(
                f"Abort tracking at frame {t}. Segmentation failed more than 5 times."
            )
            break

        if np.max(labels[t, 0] * img[t, 1]) == 0:
            print(f"Abort tracking at frame {t}. Intensity is zeros in mask.")
            break

        tmp = pd.DataFrame(
            measure.regionprops_table(
                labels[t, 0],
                img[t, 1],
                properties=(
                    "centroid",
                    "area",
                    "mean_intensity",
                    "label",
                    "moments_weighted_normalized",
                    "moments_weighted_hu",
                ),
            )
        )

        tmp["frame"] = t
        tmp["segmentation"] = method

        df.append(tmp)

    # massage the dataframe
    df = pd.concat(df, ignore_index=True)

    df["spread"] = np.sqrt(
        df["moments_weighted_normalized-2-0"] + df["moments_weighted_normalized-0-2"]
    )

    df["skew"] = np.abs(df["moments_weighted_normalized-3-0"]) + np.abs(
        df["moments_weighted_normalized-0-3"]
    )

    # add the distance to center
    df["distance_to_center"] = np.sqrt(
        np.square(df["centroid-0"] - img.shape[2] / 2)
        + np.square(df["centroid-1"] - img.shape[3] / 2)
    )

    # track the cells with trackpy
    tp.quiet()
    trj = tp.link(df, search_range=40, pos_columns=["centroid-0", "centroid-1"])

    # Keep the cell the most at the center at frame 0
    cell_to_keep = np.argmin(trj[trj["frame"] == 0]["distance_to_center"])
    trj = trj[trj["particle"] == cell_to_keep]

    # center = np.array([img.shape[2], img.shape[3]]) / 2
    # trj = link(df, center)

    # set the labels as the track id
    tracked_labels = np.zeros(labels.shape, dtype=np.uint8)
    for row in trj.iloc:
        idx = labels[int(row["frame"])] == row["label"]
        tracked_labels[int(row["frame"])][idx] = 1

    return tracked_labels, trj


def segment_and_track_dna_blobs(img, mask):
    """Analyze the blobs of DNA

    Parameters
    ----------
    img: np.ndarray
        (D,H,W) image stack
    mask: np.ndarray
        (D,H,W) mask stack

    Returns
    -------
    labels : np.ndarray
        segmented blobs labels sorted by track length
    """

    # segment the blobs
    blob = img.astype(float)
    blob = blob - ndi.gaussian_filter(blob, [5, 5, 5])
    blob = (blob > (np.median(blob) + 0.5 * blob.std())) * mask.squeeze()
    for _ in range(3):
        blob = ndi.median_filter(blob, [1, 3, 3])

    blob = np.stack([segment_watershed(b, 5.0) for b in blob])

    # get the centroids for each frame
    df = []
    for frame, labels in enumerate(blob):
        tmp = pd.DataFrame(
            measure.regionprops_table(
                labels,
                img[frame],
                properties=(
                    "centroid",
                    "mean_intensity",
                    "label",
                ),
            )
        )
        tmp["frame"] = frame
        df.append(tmp)

    df = pd.concat(df, ignore_index=True)

    # track the blobs
    tp.quiet()
    trj = tp.link(df, search_range=5, pos_columns=["centroid-0", "centroid-1"])

    # reassign the labels by order of length
    # ag = (
    #     trj.groupby("particle")["frame"]
    #     .agg(("count",))
    #     .sort_values("count", ascending=False)
    # )

    # reassign the labels by first frame
    ag = trj.groupby("particle").agg("min").sort_values("frame")
    # mapping
    tmap = {r.name: k + 1 for k, r in enumerate(ag.iloc)}

    # transform the trj dataframe
    trj["particle"] = trj["particle"].apply(lambda x: tmap[x])

    # map the labels to the tracked labels
    labels = np.zeros(blob.shape, dtype=np.uint8)
    for row in trj.iloc:
        idx = blob[int(row["frame"])] == row["label"]
        labels[int(row["frame"])][idx] = int(row["particle"])

    return labels, trj


def frame_differences(img: np.ndarray) -> np.ndarray:
    """Frame differences

    Parameters
    ----------
    img : np.ndarray
        [T,C,H,W]

    Returns
    -------
    frame difference
    """
    return np.expand_dims(np.diff(img, 1, 0), 1)


def compute_flow(img: np.ndarray, radius: int = 4):
    """Compute the flow for each frame in the sequence

    Parameters
    ----------
    img : np.ndarray
        [T,1,H,W] image sequence (1 channel)

    Returns
    -------
    img: np.ndarray
        [T,2,H,W] flow
    """
    flows = []
    for reference, moving in zip(img[:-1], img[1:]):
        flows.append(
            np.stack(
                optical_flow_ilk(
                    reference,
                    moving,
                    radius=radius,
                    gaussian=True,
                    prefilter=True,
                )
            )
        )
    flow = np.stack(flows)
    return flow


def divergence(flow: np.ndarray) -> np.ndarray:
    """Compute the divergence of the vector field

    Parameters
    ----------
    flow : np.ndarray
        [T,2,H,W]

    Returns
    -------
    Diveragence as a [T,1,H,W] np.ndarray
    """
    return np.expand_dims(
        np.gradient(flow[:, 0], axis=1)
        + np.gradient(
            flow[
                :,
                1,
            ],
            axis=2,
        ),
        axis=1,
    )


def momentum(mass: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Compute the momentum as mass x flow

    Parameters
    ----------
    mass : np.ndarray
        [T,1,H,W]
    flow : np.ndarray
        [T,2,H,W]

    Returns
    -------
    momentum as a [T,1,H,W] np.ndarray

    """
    return np.expand_dims(mass, axis=1) * flow


def average(value: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute the weighted average over time in mask"""
    return np.squeeze(
        (np.expand_dims(np.linalg.norm(value, axis=1), 1) * weights).sum(axis=(1, 2, 3))
        / weights.sum(axis=(1, 2, 3))
    )


def sum_intensity(mask, image):
    return [np.sum((mask == level) * image) for level in np.unique(mask) if level > 0]


def process(filename: str, channels=[0, 1]):
    """Process the file

    Parameters
    ----------
    filename : str
        path to the file

    Returns
    -------
    img: np.ndarray
        image as [L,C,H,W]
    cell_lbl: np.ndarray
        cell segmentation masks [L,1,H,W]
    cell_trj: pd.DataFrame
        cell tracks
    cell_flow: np.ndarray
        estimated optical flow [L,2,H,W]
    dna_lbl: np.ndarray

    dna_trj: np.ndarray
    dna_flow : np.ndarray
        estimated optical flow [L,2,H,W]

    Note
    ----
    pimg, cell_lbl, cell_trj, cell_flow, dna_lbl, dna_trj, dna_flow = process(filename)
    """

    img = tifffile.imread(filename)
    pimg = preprocess(img, 100)
    pimg2 = preprocess(img, 100, scale=[3, 0, 2, 2])
    cell_lbl, cell_trj = segment_and_track_cell(pimg2)
    # dna_diff = frame_differences(pimg[:, 1])
    cell_flow = compute_flow(pimg[:, channels[0]], 20)
    dna_flow = compute_flow(pimg[:, channels[1]], 2)
    # rho = momentum(pimg[:-1, 1], dna_flow)
    # div = divergence(rho)
    dna_lbl, dna_trj = segment_and_track_dna_blobs(pimg2[:, 1], cell_lbl)
    return pimg, cell_lbl, cell_trj, cell_flow, dna_lbl, dna_trj, dna_flow


def save_result(
    filename: str,
    name: str,
    img: np.ndarray,
    cell_lbl: np.ndarray,
    cell_trj: np.ndarray,
    cell_flow: np.ndarray,
    dna_lbl: np.ndarray,
    dna_trj: np.ndarray,
    dna_flow: np.ndarray,
):
    """Save results to a hdf5 file

    Parameters
    ----------
    filename : str
        path to the file
    name : str
        name of the group / id
    img: np.ndarray
        original image as [L,C,H,W]
    cell_lbl: np.ndarray
        segmentation masks [L,1,H,W]
    cell_trj: pd.dataframe
        positions of the cell of time [L,2]
    cell_flow:
        estimated optical flow on the cell [L,2,H,W]
    dna_lbl: np.ndarray
        segmentation masks [L,1,H,W]
    dna_trj: pd.dataframe
        positions of the cell of time [L,2]
    dna_flow:
        estimated optical flow on the cell [L,2,H,W]
    """

    with h5py.File(filename, "w") as f:
        f.create_group(name)
        f.create_dataset(f"{name}/img", data=img)
        f.create_dataset(f"{name}/cell_lbl", data=cell_lbl)
        f.create_dataset(f"{name}/cell_flow", data=cell_flow)
        f.create_dataset(f"{name}/dna_lbl", data=dna_lbl)
        f.create_dataset(f"{name}/dna_flow", data=dna_flow)

    # cell_trj and blob_trj are dataframes. Use panda to save them in the file
    cell_trj.to_hdf(filename, key=f"{name}/cell_trj", mode="a")
    dna_trj.to_hdf(filename, key=f"{name}/dna_trj", mode="a")


def inspect_result(dst):
    """Inspect the content of the result file

    Parameter
    ---------
    dst: str | Path
        Path to the result folder

    Returns
    -------
    Indices of the processed items
    """

    filelist = pd.read_csv(dst / "filelist.csv")

    return [
        row.name for row in filelist.iloc if Path(dst / f"{row.name:06d}.h5").exists()
    ]


def check_h5(folder, index):
    folder = Path(folder)
    filename = folder / f"{index:06d}.h5"
    filelist = pd.read_csv(folder / "filelist.csv")
    name = filelist["name"].iloc[index]
    with h5py.File(filename, "r") as f:
        for k in f[name]:
            print(k)


def load_result(folder: str, index: int):
    """Load the result from a HDF5 file

    Returns
    -------
    name: str
        name of the h5 group (filename)
    pimg:
        smooted image
    cell_lbl: np.ndarray
        cell segmentation mask
    cell_trj
    cell_flow
    dna_lbl
    dna_trj
    dna_flow

    """
    folder = Path(folder)
    filename = folder / f"{index:06d}.h5"
    filelist = pd.read_csv(folder / "filelist.csv")
    name = filelist["name"].iloc[index]
    with h5py.File(filename, "r") as f:
        pimg = np.array(f[name]["img"]).copy()
        cell_lbl = np.array(f[name]["cell_lbl"]).copy()
        cell_flow = np.array(f[name]["cell_flow"]).copy()
        dna_lbl = np.array(f[name]["dna_lbl"]).copy()
        dna_flow = np.array(f[name]["dna_flow"]).copy()

    dna_trj = pd.read_hdf(filename, f"{name}/dna_trj")
    cell_trj = pd.read_hdf(filename, f"{name}/cell_trj")

    return name, pimg, cell_lbl, cell_trj, cell_flow, dna_lbl, dna_trj, dna_flow


def record(
    index, filename, pimg, cell_lbl, cell_trj, cell_flow, dna_lbl, dna_trj, dna_flow
):
    """Record the results in a dataframe"""
    dna_diff = frame_differences(pimg[:, 1])
    dna_rho = momentum(pimg[:-1, 1], dna_flow)
    dna_div = divergence(dna_rho)
    df = []
    for row in cell_trj.iloc:
        frame = int(row["frame"])
        if frame < pimg.shape[0] - 1:
            mask_area = np.sum(cell_lbl[frame])
            df.append(
                {
                    "index": index,
                    "filename": filename,
                    "frame": frame,
                    "cell-x": row["centroid-1"],
                    "cell-y": row["centroid-0"],
                    "cell area": mask_area,
                    "cell flow": np.sum(
                        np.linalg.norm(cell_flow[frame], axis=0) * cell_lbl[frame]
                    )
                    / mask_area,
                    "dna frame difference": np.sum(
                        np.abs(dna_diff[frame]) * cell_lbl[frame]
                    )
                    / mask_area,
                    "dna flow": np.sum(
                        np.linalg.norm(dna_flow[frame], axis=0) * cell_lbl[frame]
                    )
                    / mask_area,
                    "dna momentum": np.sum(
                        np.linalg.norm(dna_rho[frame], axis=0) * cell_lbl[frame]
                    )
                    / mask_area,
                    "dna divergence": np.sum(np.abs(dna_div[frame]) * cell_lbl[frame])
                    / mask_area,
                    "dna intensity mean": row["mean_intensity"],
                    "dna intensity spread": row["spread"],
                    "dna intensity skew": row["skew"],
                    "dna blob count": len(dna_trj[dna_trj["frame"] == frame]),
                    "dna blob area": np.sum(dna_lbl[frame] > 0),
                    "dna blob 1 sum intensity": np.sum(
                        pimg[frame, 1] * (dna_lbl[frame] == 1)
                    ),
                    "dna blob 2 sum intensity": np.sum(
                        pimg[frame, 1] * (dna_lbl[frame] == 2)
                    ),
                    "dna blob 1 area": np.sum(dna_lbl[frame] == 1),
                    "dna blob 2 area": np.sum(dna_lbl[frame] == 2),
                }
            )
    return pd.DataFrame.from_records(df)


def split_frame(df):
    """Detect the splitting time and return the frame index

    Parameters
    ----------
    df:pd.DataFrame

    """
    k = np.where(np.array(df["cell area"]) < 0.6 * df["cell area"].max())[0]
    if len(k) > 0:
        frame = df["frame"].iloc[k[0]].item()
    else:
        frame = 0
    return frame


def create_figure(
    index,
    name,
    pimg,
    cell_lbl,
    cell_trj,
    cell_flow,
    dna_lbl,
    dna_trj,
    dna_flow,
    frame=0,
):
    """Create a figure with graphs over time

    The figure has 3 rows and 4 columns
    """

    df = record(
        index,
        name,
        pimg,
        cell_lbl,
        cell_trj,
        cell_flow,
        dna_lbl,
        dna_trj,
        dna_flow,
    )

    if frame == "auto":
        frame = split_frame(df)

    cols = df.columns[5:]

    fig, ax = plt.subplots(4, int(np.ceil((len(cols) + 1) / 4)), figsize=(12, 9))
    ax = ax.ravel()

    ax[0].imshow(uv2rgb(pimg[frame]))
    ax[0].set_axis_off()
    ax[0].set(title=f"frame:{frame}")

    for c in measure.find_contours(cell_lbl[frame, 0], 0.5):
        ax[0].plot(c[:, 1], c[:, 0], "w")

    for level in range(1, int(dna_lbl[frame].max() + 1)):
        for c in measure.find_contours((dna_lbl[frame] == level).astype(int), 0.5):
            ax[0].plot(c[:, 1], c[:, 0], "orange")

    ax[0].plot(df["cell-x"], df["cell-y"], "w", linewidth=1)

    for k, col in enumerate(cols):
        ax[k + 1].plot(df["frame"], df[col], color="#87ceeb", alpha=0.75)
        if k < 7:
            smoothed = ndi.gaussian_filter1d(df[col], sigma=5)
            ax[k + 1].plot(df["frame"], smoothed, color="#0072A0")
        if frame > 0:
            y = ax[k + 1].axis()[2], ax[k + 1].axis()[3]
            ax[k + 1].plot([frame, frame], y, color="orange", alpha=0.75)
            ax[k + 1].set(box_aspect=1, title=col, ylim=y)
        else:
            ax[k + 1].set(box_aspect=1, title=col)

    fig.set_tight_layout(True)
    plt.suptitle(f"{index}:{name}")


def figure(folder: Path, index: int, frame=0):
    """Create a figure with graphs over time

    The figure has 3 row and 4 columns

    Parameters
    ----------
    folder: Path
        result folder
    index : int
        index of the file
    frame: int | str
        frame index or "auto"

    """
    (
        name,
        pimg,
        cell_lbl,
        cell_trj,
        cell_flow,
        dna_lbl,
        dna_trj,
        dna_flow,
    ) = load_result(folder, index)

    create_figure(
        index,
        name,
        pimg,
        cell_lbl,
        cell_trj,
        cell_flow,
        dna_lbl,
        dna_trj,
        dna_flow,
        frame,
    )


def vec2rgb(x):
    """HSV code a vectorial field to RGB using HSV

    Parameters
    ----------
    x : np.ndarray
        vector field (2,H,W)
    Returns
    -------
    RGB stack (3,H,W)
    """
    h = (np.arctan2(x[0], x[1]) + math.pi) / math.tau
    v = np.linalg.norm(x, axis=0)
    y = np.stack((h, v), -1).reshape(x.shape[1] * x.shape[2], 2)
    return np.clip(
        np.stack([colorsys.hsv_to_rgb(hv[0], hv[1], 1) for hv in y], 0).reshape(
            [x.shape[1], x.shape[2], 3]
        ),
        0,
        1,
    )


def create_strip(
    index,
    name,
    pimg,
    cell_lbl,
    cell_trj,
    cell_flow,
    dna_lbl,
    dna_trj,
    dna_flow,
    colormap="Greys",
    selection=None,
    quiver=False,
):
    """Create a strip visualization"""

    # Select frame around the split frame
    if selection == "auto":
        df = record(
            index,
            name,
            pimg,
            cell_lbl,
            cell_trj,
            cell_flow,
            dna_lbl,
            dna_trj,
            dna_flow,
        )
        frame = split_frame(df)
        start_frame = max(0, frame - 100)
        end_frame = min(pimg.shape[0] - 1, frame + 100)
        selection = slice(start_frame, end_frame, 10)

    dna_diff = frame_differences(pimg[:, 1])
    dna_rho = momentum(pimg[:-1, 1], dna_flow)
    dna_div = divergence(dna_rho)
    s = 2
    x, y = np.meshgrid(
        *[np.arange(0, n, s) for n in [pimg.shape[2], pimg.shape[3]]], indexing="xy"
    )
    X, Y = np.meshgrid(
        *[np.arange(0, n) for n in [pimg.shape[2], pimg.shape[3]]], indexing="xy"
    )
    if selection is None:
        indices = np.arange(0, pimg.shape[0], 20)
    else:
        indices = np.arange(selection.start, selection.stop, selection.step)
    fig, ax = plt.subplots(5, len(indices), figsize=(len(indices), 5))
    dmax = (np.abs(dna_diff) * cell_lbl[:-1]).max() / 2
    vmax = (np.linalg.norm(dna_flow, axis=1)).max() / 2
    # rmax = (np.linalg.norm(rho, axis=1)).max() / 2
    drmax = (np.abs(dna_div) * cell_lbl[:-1]).max() / 2
    for k, n in enumerate(indices):
        if n >= pimg.shape[0]:
            break
        ax[0, k].imshow(uv2rgb(pimg[n]))
        ax[0, k].set_axis_off()
        for c in measure.find_contours(cell_lbl[n, 0], 0.5):
            ax[0, k].plot(c[:, 1], c[:, 0], "white", alpha=0.8)
        for level in range(1, int(dna_lbl[n].max() + 1)):
            for c in measure.find_contours((dna_lbl[n] == level).astype(int), 0.5):
                ax[0, k].plot(c[:, 1], c[:, 0], "#FFA500", alpha=0.8)
        ax[0, k].set(title=f"{n}")
        ax[1, k].imshow(
            (dna_diff[n] * cell_lbl[n]).squeeze(), vmin=-dmax, vmax=dmax, cmap=colormap
        )
        ax[1, k].set_axis_off()
        ax[1, 0].text(-10, pimg.shape[2] / 2, "diff", rotation=90)

        ax[2, k].imshow(vec2rgb(dna_flow[n] / vmax))
        if quiver:
            ax[2, k].quiver(
                x,
                y,
                dna_flow[n, 1, ::s, ::s],
                dna_flow[n, 0, ::s, ::s],
                color="k",
                units="dots",
                angles="xy",
                scale_units="xy",
                lw=3,
            )
        else:
            ax[2, k].streamplot(
                X,
                Y,
                dna_flow[n, 1],
                dna_flow[n, 0],
                density=1,
                linewidth=0.5,
                arrowsize=0.1,
                color="k",
            )
        ax[2, k].set_axis_off()
        ax[2, 0].text(-10, pimg.shape[2] / 2, "flow", rotation=90)

        ax[3, k].imshow(
            np.ones([pimg.shape[2], pimg.shape[3]]), cmap="gray", vmin=0, vmax=1
        )
        ax[3, k].imshow(vec2rgb(dna_rho[n]))
        if quiver:
            ax[4, k].quiver(
                x,
                y,
                dna_rho[n, 1, ::s, ::s],
                dna_rho[n, 0, ::s, ::s],
                color="k",
                units="dots",
                angles="xy",
                scale_units="xy",
                lw=3,
            )
        else:
            ax[3, k].streamplot(
                X,
                Y,
                dna_rho[n, 1],
                dna_rho[n, 0],
                density=1,
                linewidth=0.5,
                arrowsize=0.1,
                color="k",
            )
        ax[3, k].set_axis_off()
        ax[3, 0].text(-10, pimg.shape[2] / 2, "mom", rotation=90)

        ax[4, k].imshow(
            (dna_div[n] * cell_lbl[n]).squeeze(), vmin=-drmax, vmax=drmax, cmap=colormap
        )
        ax[4, k].set_axis_off()
        ax[4, 0].text(-10, pimg.shape[2] / 2, "div", rotation=90)

    fig.suptitle(f"{index}:{Path(name).stem}")
    plt.subplots_adjust(wspace=0, hspace=0)


def strip(
    folder,
    index,
    colormap="Greys",
    selection=None,
    quiver=False,
):
    """Create a strip illustration with vignette etc"""

    (
        name,
        pimg,
        cell_lbl,
        cell_trj,
        cell_flow,
        dna_lbl,
        dna_trj,
        dna_flow,
    ) = load_result(folder, index)

    create_strip(
        index,
        name,
        pimg,
        cell_lbl,
        cell_trj,
        cell_flow,
        dna_lbl,
        dna_trj,
        dna_flow,
        colormap,
        selection,
        quiver,
    )


def make_vector(data, step: int = 2):
    """Create a vector array for napari visualization

    Parameters
    ----------
    data: np.ndarray
        input array [T,2,W,D]
    step: int
        sub sampling

    Returns
    -------
    Array

    """
    step = 2
    x, y = np.meshgrid(
        *[np.arange(0, n, step) for n in [data.shape[2], data.shape[3]]], indexing="xy"
    )
    return np.concatenate(
        [
            np.stack(
                (
                    np.stack((k * np.ones(x.size), y.ravel(), x.ravel()), 1),
                    np.stack(
                        (
                            np.zeros(x.size),
                            (f[0, ::step, ::step]).ravel(),
                            (f[1, ::step, ::step]).ravel(),
                        ),
                        1,
                    ),
                ),
                1,
            )
            for k, f in enumerate(data)
        ]
    )


def list_files(root: Path, dst: Path):
    """List the files in the source folders and add them to a csv file

    Parameters
    ----------
    root: Path
        root of the path to the files
    dst: Path
        path to the result folfer

    Returns
    -------
    list of files
    """

    dst.mkdir(exist_ok=True)

    filelist = pd.DataFrame.from_records(
        [
            {"path": x.relative_to(root), "name": x.stem, "condition": "unknown"}
            for x in root.rglob("Crop*/[!.]*.tif")
        ]
    )

    print(f"Number of files {len(filelist)}")
    opath = dst / "filelist.csv"
    filelist.to_csv(opath)
    return filelist


def get_figure_data():
    """Figure as a numpy array"""
    with io.BytesIO() as buff:
        plt.savefig(buff, format="raw")
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = plt.gcf().canvas.get_width_height()
    return data.reshape((int(h), int(w), -1))


def process_file(root: Path, dst: Path, index: int):
    """Process a single item from the filelist.csv stored in the dest folder

    Parameters
    ----------
    root: Path
        root of the path to the files
    dst: Path
        path to the result folder containing a `filelist.csv`

    """

    # Determine the file name from the list of files in the dst folder
    if not (dst / "filelist.csv").exists():
        raise (f"The folder {dst} has no filelist.csv. Run the 'list' command first.")

    filelist = pd.read_csv(dst / "filelist.csv")
    filename = Path(filelist["path"].iloc[index])
    ipath = root / filename

    if not ipath.exists():
        print(f"filepath '{ipath}' does not exist")
        exit(1)
    else:
        print(f"filepath '{ipath}'")

    # process the file
    pimg, cell_lbl, cell_trj, cell_flow, dna_lbl, dna_trj, dna_flow = process(ipath)

    # Save the results as a hdf5 file
    h5_path = dst / f"{index:06d}.h5"
    print(f"Saving h5 {h5_path}")
    if h5_path.exists():
        h5_path.unlink()

    save_result(
        h5_path,
        filename.stem,
        pimg,
        cell_lbl,
        cell_trj,
        cell_flow,
        dna_lbl,
        dna_trj,
        dna_flow,
    )

    # export data as csv
    df = record(
        index,
        filename,
        pimg,
        cell_lbl,
        cell_trj,
        cell_flow,
        dna_lbl,
        dna_trj,
        dna_flow,
    )
    csv_path = dst / f"{index:06d}.csv"
    print(f"Saving csv file {csv_path}")
    df.to_csv(csv_path)

    # create a strip visualization
    create_strip(
        filename.stem,
        pimg,
        cell_lbl,
        cell_trj,
        cell_flow,
        dna_lbl,
        dna_trj,
        dna_flow,
        "Greys",
        selection="auto",
        quiver=False,
    )
    # save the figure in a numpy array
    strip_data = get_figure_data()

    # create a figure with intensities over time
    create_figure(
        index,
        filename.name,
        pimg,
        cell_lbl,
        cell_trj,
        cell_flow,
        dna_lbl,
        dna_trj,
        dna_flow,
        frame="auto",
    )
    fig_data = get_figure_data()

    # concaternate both arrays
    data = 255 * np.ones(
        [
            strip_data.shape[0] + fig_data.shape[0],
            max(strip_data.shape[1], fig_data.shape[1]),
            4,
        ],
        dtype=np.uint8,
    )
    data[
        : strip_data.shape[0],
        data.shape[1] // 2 - strip_data.shape[1] // 2 : data.shape[1] // 2
        + strip_data.shape[1] // 2,
        :,
    ] = strip_data
    data[
        strip_data.shape[0] : strip_data.shape[0] + fig_data.shape[0],
        data.shape[1] // 2 - fig_data.shape[1] // 2 : data.shape[1] // 2
        + fig_data.shape[1] // 2,
        :,
    ] = fig_data
    fig_path = dst / f"{index:06d}.jpg"
    plt.imsave(fig_path, data)


def _process_file(args):
    """Process file as a command line"""
    process_file(Path(args.root), Path(args.dst), args.index)


def _list_files(args):
    """List files as a command line"""
    list_files(Path(args.root), Path(args.dst))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("dna-sufolobus")
    subparsers = parser.add_subparsers()

    subparser_list = subparsers.add_parser("list")
    subparser_list.add_argument(
        "--root", type=Path, required=False, help="path to the source data"
    )
    subparser_list.add_argument(
        "--dst", type=Path, required=True, help="path to the destination result"
    )
    subparser_list.set_defaults(func=_list_files)

    subparser_process = subparsers.add_parser("process")
    subparser_process.add_argument(
        "--root", type=Path, required=False, help="root source data"
    )
    subparser_process.add_argument(
        "--dst", type=Path, required=True, help="path to the destination result"
    )
    subparser_process.add_argument(
        "--index", type=int, required=True, help="index of the filelist"
    )
    subparser_process.set_defaults(func=_process_file)
    args = parser.parse_args()
    args.func(args)
