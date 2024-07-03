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


def segment_watershed(frame: np.ndarray):
    m = (frame[0] + frame[1]) > 5
    d = ndi.gaussian_filter(ndi.distance_transform_edt(m), 4) * m
    seed = ndi.label(d == ndi.maximum_filter(d, 5))[0] * m
    labels = segmentation.watershed(-d, seed) * m
    return labels


def preprocess(img: np.ndarray, background: float):
    """Preprocess the image sequence with a gaussian blur"""
    return np.maximum(
        ndi.median_filter(
            ndi.gaussian_filter(img.astype(float) - background, [1, 0, 1, 1]),
            [3, 1, 1, 1],
        ),
        0,
    )


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
            labels = segment_watershed(frame)

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

    for t, frame in enumerate(img):
        # segment the cells
        labels[t, 0] = model.eval(frame, diameter=22)[0]

        if labels[t, 0].max() == 0:
            labels[t, 0] = segment_watershed(frame)

        if labels[t, 0].max() == 0:
            if t > 0:
                labels[t, 0] = labels[t - 1, 0]

        # measure centroids
        tmp = pd.DataFrame(
            measure.regionprops_table(
                labels[t, 0],
                img[t, 0],
                properties=("centroid", "mean_intensity", "label"),
            )
        )
        tmp["frame"] = t

        df.append(tmp)

    # massage the dataframe
    df = pd.concat(df, ignore_index=True).rename(
        columns={"centroid-0": "y", "centroid-1": "x", "mean_intensity": "mass"}
    )

    # add the distance to center
    df["distance to center"] = np.sqrt(
        np.square(df["x"] - img.shape[-1] / 2) + np.square(df["y"] - img.shape[-2] / 2)
    )

    # track the cells
    tp.quiet()

    trj = tp.link(df, 10, memory=2)

    # Keep the cell the most at the center at frame 0
    cell_to_keep = np.argmin(trj[trj["frame"] == 0]["distance to center"])
    trj = trj[trj["particle"] == cell_to_keep]

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
    blob = img - ndi.gaussian_filter(img, [1, 5, 5])
    blob = (blob > (np.median(blob) + 0.5 * blob.std())) * mask.squeeze()
    blob = ndi.median_filter(blob, [3, 5, 5])
    blob = np.stack([ndi.label(b)[0] for b in blob])
    df = []
    for frame, labels in enumerate(blob):
        tmp = pd.DataFrame(
            measure.regionprops_table(
                labels,
                img[frame],
                properties=("centroid", "mean_intensity", "label"),
            )
        )
        tmp["frame"] = frame
        df.append(tmp)
    df = pd.concat(df, ignore_index=True).rename(
        columns={"centroid-0": "y", "centroid-1": "x", "mean_intensity": "mass"}
    )
    tp.quiet()
    trj = tp.link(df, 10, memory=2)
    ag = (
        trj.groupby("particle")["particle"]
        .agg(("count",))
        .sort_values("count", ascending=False)
    )
    labels = np.zeros(blob.shape)
    for row in trj.iloc:
        idx = blob[int(row["frame"])] == row["label"]
        labels[int(row["frame"])][idx] = ag.index[int(row["particle"])] + 1

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


def compute_flow(img: np.ndarray):
    """Compute the flow for each frame in the sequence
    Parameters
    ----------
    img : np.ndarray
        [T,1,H,W]
    """
    flows = []
    for reference, moving in zip(img[:-1], img[1:]):
        flows.append(
            np.stack(
                optical_flow_ilk(
                    reference,
                    moving,
                    radius=4,
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
    divergence  [T,1,H,W]
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
    """Compute the momentum as mass x flow"""
    return np.expand_dims(mass, axis=1) * flow


def average(value: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute the weighted average over time in mask"""
    return np.squeeze(
        (np.expand_dims(np.linalg.norm(value, axis=1), 1) * weights).sum(axis=(1, 2, 3))
        / weights.sum(axis=(1, 2, 3))
    )


def sum_intensity(mask, image):
    return [np.sum((mask == level) * image) for level in np.unique(mask) if level > 0]


def dna_blob_metrics(blob, img):
    """Quantify dna blob intensity along time

    Parameters
    ----------
    blob: np.ndarray
        labels (D,W,H) for each time point
    img: np.ndarray
        intensity (D,W,H)

    Returns
    -------
    count: np.ndarray
        number of blobs
    area: np.ndarray
        area of the two biggest blobs
    asymmetry_area: np.ndarray
        area ratio of the smallest vs the sum of the two blobs
    intensity: np.ndarray
        sum of intensity of the two biggest blobs
    asymmetry_int: np.ndarray
        intensity ratio of the smallest vs the sum of the two blobs
    """
    count = np.zeros(img.shape[0])
    area0 = np.zeros(blob.shape[0])
    area1 = np.zeros(blob.shape[0])
    intensity0 = np.zeros(blob.shape[0])
    intensity1 = np.zeros(blob.shape[0])

    for k, labels in enumerate(blob):
        n = len(np.unique(labels)) - 1
        count[k] = n
        p = measure.regionprops_table(
            labels,
            img[k],
            properties=(
                "area",
                "mean_intensity",
                "centroid",
            ),
            extra_properties=(sum_intensity,),
        )
        if n == 1:
            area0[k] = p["area"][0]
            intensity0[k] = p["sum_intensity-0"][0]
        if n >= 2:
            perm = np.argsort(p["area"])[::-1]
            area[k] = p["area"][perm[0]] + p["area"][perm[1]]
            asymmetry_area[k] = p["area"][perm[1]] / area[k]
            intensity[k] = p["sum_intensity-0"][perm[0]] + p["sum_intensity-0"][perm[1]]
            asymmetry_int[k] = p["sum_intensity-0"][perm[1]] / intensity[k]
    return count, area, asymmetry_area, intensity, asymmetry_int


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
    flow = compute_flow(pimg[:, 1])
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
    with h5py.File(filename, "a") as f:
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
                    "cell-x": row["x"],
                    "cell-y": row["y"],
                    "cell-area": mask_area,
                    "diff": np.sum(diff[frame] * cell_mask[frame])
                    / np.sum(cell_mask[frame]),
                    "flow": np.sum(
                        np.linalg.norm(flow[frame], axis=0) * cell_mask[frame]
                    )
                    / mask_area,
                    "momentum": np.sum(
                        np.linalg.norm(rho[frame], axis=0) * cell_mask[frame]
                    )
                    / mask_area,
                    "divergence": np.sum(div[frame] * cell_mask[frame]) / mask_area,
                    "dna mean intensity": np.sum(img[frame, 1] * cell_mask[frame])
                    / mask_area,
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


def figure(
    ax,
    k,
    cond,
    name,
    img,
    mask,
    position,
    speed,
    diff,
    flow,
    rho,
    div,
    blob,
    title=True,
):
    """Create a figure with graphs over time"""

    count, area, asym_area, intensity, asym_int = dna_blob_metrics(blob, img[:, 1])

    ax[0].imshow(uv2rgb(img[0]))
    ax[0].set_axis_off()
    for c in measure.find_contours(mask[0, 0].astype(int), 0.5):
        ax[0].plot(c[:, 1], c[:, 0], "w")
    ax[0].plot(position[:, 1], position[:, 0], "w", linewidth=1)
    ax[0].text(-22, img.shape[2] / 2 + 10, f"{k}:{cond}", fontsize=12, rotation=90)
    ax[0].text(-10, img.shape[2] / 2 + 10, Path(name).stem, fontsize=6, rotation=90)

    ax[1].plot(average(np.expand_dims(img[:-1, 1], 1), mask[:-1]))
    if title:
        ax[1].set(box_aspect=1, title="dna mean intensity", ylim=100)
    else:
        ax[1].set(box_aspect=1, ylim=100)

    ax[2].plot(average(diff, mask[:-1]))
    if title:
        ax[2].set(box_aspect=1, title="diff", ylim=0)
    else:
        ax[2].set(box_aspect=1, ylim=0)

    ax[3].plot(average(flow, mask[:-1]))
    if title:
        ax[3].set(box_aspect=1, title="flow", ylim=0)
    else:
        ax[3].set(box_aspect=1, ylim=0)

    ax[4].plot(average(rho, mask[:-1]))
    if title:
        ax[4].set(box_aspect=1, title="momentum", ylim=0)
    else:
        ax[4].set(box_aspect=1, ylim=0)

    ax[5].plot(average(div, mask[:-1]))
    if title:
        ax[5].set(box_aspect=1, title="divergence", ylim=0)
    else:
        ax[5].set(box_aspect=1, ylim=0)

    ax[6].plot(count[:-1])
    if title:
        ax[6].set(box_aspect=1, title="count", ylim=0)
    else:
        ax[6].set(box_aspect=1, ylim=0)

    ax[7].plot(area[:-1])
    if title:
        ax[7].set(box_aspect=1, title="dna blob area", ylim=0)
    else:
        ax[7].set(box_aspect=1, ylim=0)

    ax[8].plot(asym_area[:-1])
    if title:
        ax[8].set(box_aspect=1, title="dna blob area asymmetry", ylim=0)
    else:
        ax[8].set(box_aspect=1, ylim=0)

    ax[9].plot(intensity[:-1])
    if title:
        ax[9].set(box_aspect=1, title="dna blob sum intenity", ylim=0)
    else:
        ax[9].set(box_aspect=1, ylim=0)

    ax[10].plot(asym_int[:-1])
    if title:
        ax[10].set(
            box_aspect=1,
            title="dna blob intenity asymmetry",
            ylim=0,
        )
    else:
        ax[10].set(box_aspect=1, ylim=0)


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
    name,
    img,
    mask,
    position,
    speed,
    diff,
    flow,
    rho,
    div,
    blob,
    colormap="Greys",
    selection=None,
    quiver=False,
):
    """Create a strip"""

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
    dmax = (np.abs(diff) * mask[:-1]).max() / 2
    vmax = (np.linalg.norm(flow, axis=1)).max() / 2
    rmax = (np.linalg.norm(rho, axis=1)).max() / 2
    drmax = (np.abs(div) * mask[:-1]).max() / 2
    for k, n in enumerate(indices):
        ax[0, k].imshow(uv2rgb(img[n]))
        ax[0, k].set_axis_off()
        for c in measure.find_contours(mask[n, 0], 0.5):
            ax[0, k].plot(c[:, 1], c[:, 0], "white", alpha=0.8)
        for c in measure.find_contours(blob[n] > 0, 0.5):
            ax[0, k].plot(c[:, 1], c[:, 0], "#FFA500", alpha=0.8)
        ax[0, k].set(title=f"{n}")
        ax[1, k].imshow(
            (diff[n] * mask[n]).squeeze(), vmin=-dmax, vmax=dmax, cmap=colormap
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
            (div[n] * mask[n]).squeeze(), vmin=-drmax, vmax=drmax, cmap=colormap
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
