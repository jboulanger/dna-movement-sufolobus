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


def preprocess(img: np.ndarray, background: float):
    """Preprocess the image sequence with a gaussian blur"""
    return np.maximum(
        ndi.gaussian_filter(img.astype(float) - background, [3, 0, 1, 1]),
        0,
    )

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


def segment_and_track_cell(img: np.ndarray):
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

    # # track the cells with trackpy
    # tp.quiet()
    # trj = tp.link(df, 20, pos_columns=["centroid-0", "centroid-1"])

    # # Keep the cell the most at the center at frame 0
    # cell_to_keep = np.argmin(trj[trj["frame"] == 0]["distance_to_center"])
    # trj = trj[trj["particle"] == cell_to_keep]

    center = np.array([img.shape[2], img.shape[3]]) / 2
    trj = link(df, center)

    # set the labels as the track id
    tracked_labels = np.zeros(labels.shape)
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

    df = pd.concat(df, ignore_index=True).rename(
        columns={"centroid-0": "y", "centroid-1": "x", "mean_intensity": "mass"}
    )

    # track the blobs
    tp.quiet()
    trj = tp.link(df, 5)

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
    labels = np.zeros(blob.shape)
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


def process(filename: str):
    """Process the file

    Parameters
    ----------
    filename : str
        path to the file

    Returns
    -------
    img: np.ndarray
        original image as [L,C,H,W]
    mask: np.ndarray
        segmentation masks [L,1,H,W]
    position: np.ndarray
        positions of the cell of time [L,2]
    speed: np.ndarray
        speed of the cell of time [L,2]
    diff: np.ndarray
        frame difference [L,1,H,W]
    flow: np.ndarray
        estimated optical flow corrected from the cell speed [L,2,H,W]
    rho: np.ndarray
        momentum rho.v [L,2,H,W]
    div: np.ndarray
        divergence of the momentum (div(rho.v))[L,1,H,W]

    """

    img = tifffile.imread(filename)
    pimg = preprocess(img, 100)
    cell_mask, cell_trj = segment_and_track_cell(pimg)
    diff = frame_differences(pimg[:, 1])
    flow_cells = compute_flow(pimg[:, 0], 10)
    flow = compute_flow(pimg[:, 1]) - flow_cells
    rho = momentum(pimg[:-1, 1], flow)
    div = divergence(rho)
    blob_labels, blobs_trj = segment_and_track_dna_blobs(pimg[:, 1], cell_mask)
    return img, cell_mask, cell_trj, diff, flow, rho, div, blob_labels, blobs_trj


def save_result(
    filename: str,
    name: str,
    img: np.ndarray,
    cell_mask: np.ndarray,
    cell_trj: np.ndarray,
    diff: np.ndarray,
    flow: np.ndarray,
    rho: np.ndarray,
    div: np.ndarray,
    blob_labels: np.ndarray,
    blob_trj: np.ndarray,
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
    cell_mask: np.ndarray
        segmentation masks [L,1,H,W]
    cell_trj: pd.dataframe
        positions of the cell of time [L,2]
    speed: np.ndarray
        speed of the cell of time [L,2]
    diff: np.ndarray
        frame difference [L,1,H,W]
    flow: np.ndarray
        estimated optical flow corrected from the cell speed [L,2,H,W]
    rho: np.ndarray
        momentum rho.v [L,2,H,W]
    div: np.ndarray
        divergence of the momentum (div(rho.v))[L,1,H,W]
    blob_labels: np.ndarray
        mask of the segmented DNA blobs
    blob_trj: pd.dataframe
    """
    sname = Path(name).stem

    with h5py.File(filename, "w") as f:
        f.create_group(sname)
        f.create_dataset(f"{sname}/img", data=img)
        f.create_dataset(f"{sname}/cell_mask", data=cell_mask)
        f.create_dataset(f"{sname}/diff", data=diff)
        f.create_dataset(f"{sname}/flow", data=flow)
        f.create_dataset(f"{sname}/rho", data=rho)
        f.create_dataset(f"{sname}/div", data=div)
        f.create_dataset(f"{sname}/blob_labels", data=blob_labels)

    cell_trj.to_hdf(filename, key=f"{sname}/cell_trj", mode="a")
    blob_trj.to_hdf(filename, key=f"{sname}/blob_trj", mode="a")


def inspect_result(filename):
    """Inspect the content of the result file

    Parameter
    ---------
    filename: str | Path
        Path to the result HDF5 file

    Returns
    -------
    HDF5 groups names corresponding to the processed items
    """
    groups = None
    with h5py.File(filename, "r") as f:
        groups = [k for k in f]
        # for k1 in f:
        #     for k2 in f[k1]:
        #         print(k2)
    return groups


def load_result(filename: str, name: str):
    """Load the result from a HDF5 file"""
    with h5py.File(filename, "r") as f:
        img = np.array(f[name]["img"]).copy()
        cell_mask = np.array(f[name]["cell_mask"]).copy()
        diff = np.array(f[name]["diff"]).copy()
        flow = np.array(f[name]["flow"]).copy()
        rho = np.array(f[name]["rho"]).copy()
        div = np.array(f[name]["div"]).copy()
        blob_labels = np.array(f[name]["blob_labels"]).copy()
    blob_trj = pd.read_hdf(filename, f"{name}/blob_trj")
    cell_trj = pd.read_hdf(filename, f"{name}/cell_trj")
    return img, cell_mask, cell_trj, diff, flow, rho, div, blob_labels, blob_trj


def record(
    filename, img, cell_mask, cell_trj, diff, flow, rho, div, blob_labels, blob_trj
):
    """Record the results in a dataframe"""

    df = []
    for row in cell_trj.iloc:
        frame = int(row["frame"])
        if frame < img.shape[0] - 1:
            mask_area = np.sum(cell_mask[frame])
            df.append(
                {
                    "filename": filename,
                    "frame": frame,
                    "cell-x": row["centroid-1"],
                    "cell-y": row["centroid-0"],
                    "cell-area": mask_area,
                    "frame difference": np.sum(np.abs(diff[frame]) * cell_mask[frame])
                    / np.sum(cell_mask[frame]),
                    "flow": np.sum(
                        np.linalg.norm(flow[frame], axis=0) * cell_mask[frame]
                    )
                    / mask_area,
                    "momentum": np.sum(
                        np.linalg.norm(rho[frame], axis=0) * cell_mask[frame]
                    )
                    / mask_area,
                    "divergence": np.sum(np.abs(div[frame]) * cell_mask[frame])
                    / mask_area,
                    "dna intensity mean": row["mean_intensity"],
                    "dna intensity spread": row["spread"],
                    "dna intensity skew": row["skew"],
                    "dna blob count": len(blob_trj[blob_trj["frame"] == frame]),
                    "dna blob area": np.sum(blob_labels[frame] > 0),
                    "dna blob 1 sum intensity": np.sum(
                        img[frame, 1] * (blob_labels[frame] == 1)
                    ),
                    "dna blob 2 sum intensity": np.sum(
                        img[frame, 1] * (blob_labels[frame] == 2)
                    ),
                    "dna blob 1 area": np.sum(blob_labels[frame] == 1),
                    "dna blob 2 area": np.sum(blob_labels[frame] == 2),
                }
            )
    return pd.DataFrame.from_records(df)


def split_frame(df):
    k = np.where(np.array(df["cell-area"]) < df["cell-area"].mean())[0]
    if len(k) > 0:
        frame = df["frame"].iloc[k[0]].item()
    else:
        frame = 0
    return frame


def figure(filename, name, frame=0):
    """Create a figure with graphs over time
    The figure has xx columns
    """

    img, cell_mask, cell_trj, diff, flow, rho, div, blob_labels, blob_trj = load_result(
        filename, name
    )

    df = record(
        filename,
        img,
        cell_mask,
        cell_trj,
        diff,
        flow,
        rho,
        div,
        blob_labels,
        blob_trj,
    )

    if frame == "auto":
        k = split_frame(df)

    cols = df.columns[5:-2]

    fig, ax = plt.subplots(3, (len(cols) + 1) // 3, figsize=(12, 9))
    ax = ax.ravel()
    ax[0].imshow(uv2rgb(img[frame]))
    ax[0].set_axis_off()
    ax[0].set(title=f"frame:{frame}")

    for c in measure.find_contours(cell_mask[frame, 0], 0.5):
        ax[0].plot(c[:, 1], c[:, 0], "w")

    for level in range(1, int(blob_labels[frame].max() + 1)):
        for c in measure.find_contours((blob_labels[frame] == level).astype(int), 0.5):
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


def strip(
    filename,
    name,
    colormap="Greys",
    selection=None,
    quiver=False,
):
    """Create a strip"""
    img, cell_mask, cell_trj, diff, flow, rho, div, blob_labels, blob_trj = load_result(
        filename, name
    )
    s = 2
    x, y = np.meshgrid(
        *[np.arange(0, n, s) for n in [img.shape[2], img.shape[3]]], indexing="xy"
    )
    X, Y = np.meshgrid(
        *[np.arange(0, n) for n in [img.shape[2], img.shape[3]]], indexing="xy"
    )
    if selection is None:
        indices = np.arange(0, img.shape[0], 20)
    else:
        indices = np.arange(selection.start, selection.stop, selection.step)
    fig, ax = plt.subplots(5, len(indices), figsize=(len(indices), 5))
    dmax = (np.abs(diff) * cell_mask[:-1]).max() / 2
    vmax = (np.linalg.norm(flow, axis=1)).max() / 2
    rmax = (np.linalg.norm(rho, axis=1)).max() / 2
    drmax = (np.abs(div) * cell_mask[:-1]).max() / 2
    for k, n in enumerate(indices):
        if n >= img.shape[0]:
            break
        ax[0, k].imshow(uv2rgb(img[n]))
        ax[0, k].set_axis_off()
        for c in measure.find_contours(cell_mask[n, 0], 0.5):
            ax[0, k].plot(c[:, 1], c[:, 0], "white", alpha=0.8)
        for level in range(1, int(blob_labels[n].max() + 1)):
            for c in measure.find_contours((blob_labels[n] == level).astype(int), 0.5):
                ax[0, k].plot(c[:, 1], c[:, 0], "#FFA500", alpha=0.8)
        ax[0, k].set(title=f"{n}")
        ax[1, k].imshow(
            (diff[n] * cell_mask[n]).squeeze(), vmin=-dmax, vmax=dmax, cmap=colormap
        )
        ax[1, k].set_axis_off()
        ax[1, 0].text(-10, img.shape[2] / 2, "diff", rotation=90)

        ax[2, k].imshow(vec2rgb(flow[n] / vmax))
        if quiver:
            ax[2, k].quiver(
                x,
                y,
                flow[n, 1, ::s, ::s],
                flow[n, 0, ::s, ::s],
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
                flow[n, 1],
                flow[n, 0],
                density=1,
                linewidth=0.5,
                arrowsize=0.1,
                color="k",
            )
        ax[2, k].set_axis_off()
        ax[2, 0].text(-10, img.shape[2] / 2, "flow", rotation=90)

        ax[3, k].imshow(
            np.ones([img.shape[2], img.shape[3]]), cmap="gray", vmin=0, vmax=1
        )
        ax[3, k].imshow(vec2rgb(rho[n]))
        if quiver:
            ax[4, k].quiver(
                x,
                y,
                rho[n, 1, ::s, ::s],
                rho[n, 0, ::s, ::s],
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
                rho[n, 1],
                rho[n, 0],
                density=1,
                linewidth=0.5,
                arrowsize=0.1,
                color="k",
            )
        ax[3, k].set_axis_off()
        ax[3, 0].text(-10, img.shape[2] / 2, "mom", rotation=90)

        ax[4, k].imshow(
            (div[n] * cell_mask[n]).squeeze(), vmin=-drmax, vmax=drmax, cmap=colormap
        )
        ax[4, k].set_axis_off()
        ax[4, 0].text(-10, img.shape[2] / 2, "div", rotation=90)

    fig.suptitle(Path(name).stem)
    plt.subplots_adjust(wspace=0, hspace=0)


def make_vector(data, step=2):
    """Create a vector array for napari visualization"""
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
