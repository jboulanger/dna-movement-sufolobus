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
        ndi.gaussian_filter(img.astype(float) - background, [3, 0, 1, 1]), 0
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


def dna_blob(img, mask):
    """Analyze the blobs of DNA

    Parameters
    ----------
    img: np.ndarray
        (D,H,W) image stack
    mask: np.ndarray
        (D,H,W) mask stack

    Returns
    -------
    blob : np.ndarray
        segmented blobs labels
    """
    blob = img - ndi.gaussian_filter(img, [1, 5, 5])
    blob = (blob > (np.median(blob) + 0.5 * blob.std())) * mask.squeeze()
    blob = ndi.median_filter(blob, [3, 5, 5])
    labels = np.stack([ndi.label(b)[0] for b in blob])
    return labels


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
        suim of intensity of the two biggest blobs
    asymmetry_int: np.ndarray
        intensity ratio of the smallest vs the sum of the two blobs
    """
    asymmetry_area = np.zeros(blob.shape[0])
    asymmetry_int = np.zeros(blob.shape[0])
    count = np.zeros(img.shape[0])
    area = np.zeros(img.shape[0])
    intensity = np.zeros(img.shape[0])
    for k, labels in enumerate(blob):
        n = len(np.unique(labels)) - 1
        count[k] = n
        p = measure.regionprops_table(
            labels, img[k], properties=("area", "mean_intensity")
        )
        p["sum_intensity"] = p["area"] * p["mean_intensity"]
        if n == 1:
            area[k] = p["area"][0]
            intensity[k] = p["sum_intensity"][0]
        if n >= 2:
            perm = np.argsort(p["area"])[::-1]
            area[k] = p["area"][perm[0]] + p["area"][perm[1]]
            asymmetry_area[k] = p["area"][perm[1]] / (
                p["area"][perm[0]] + p["area"][perm[1]]
            )
            intensity[k] = p["sum_intensity"][perm[0]] + p["sum_intensity"][perm[0]]
            asymmetry_int[k] = p["sum_intensity"][perm[1]] / (
                p["sum_intensity"][perm[0]] + p["sum_intensity"][perm[0]]
            )
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
    mask, position = segment_and_track(pimg)
    speed = np.diff(position, 1, axis=0)
    diff = frame_differences(pimg[:, 1])
    flow = compute_flow(pimg[:, 1])
    rho = momentum(pimg[:-1, 1], flow)
    div = divergence(rho)
    blob = dna_blob(pimg[:, 1], mask)
    return img, mask, position, speed, diff, flow, rho, div, blob


def save_result(
    filename: str,
    name: str,
    img: np.ndarray,
    mask: np.ndarray,
    position: np.ndarray,
    speed: np.ndarray,
    diff: np.ndarray,
    flow: np.ndarray,
    rho: np.ndarray,
    div: np.ndarray,
    blob: np.ndarray,
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
    blob: np.ndarray
        mask of the segmented DNA blobs
    """
    sname = Path(name).stem
    with h5py.File(filename, "a") as f:
        f.create_group(sname)
        f.create_dataset(f"{sname}/img", data=img)
        f.create_dataset(f"{sname}/mask", data=mask)
        f.create_dataset(f"{sname}/position", data=position)
        f.create_dataset(f"{sname}/speed", data=speed)
        f.create_dataset(f"{sname}/diff", data=diff)
        f.create_dataset(f"{sname}/flow", data=flow)
        f.create_dataset(f"{sname}/rho", data=rho)
        f.create_dataset(f"{sname}/div", data=div)
        f.create_dataset(f"{sname}/blob", data=blob)


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
    return groups


def load_result(filename: str, name: str):
    """Load the result from a HDF5 file"""
    with h5py.File(filename, "r") as f:
        img = np.array(f[name]["img"]).copy()
        mask = np.array(f[name]["mask"]).copy()
        position = np.array(f[name]["position"]).copy()
        speed = np.array(f[name]["speed"]).copy()
        diff = np.array(f[name]["diff"]).copy()
        flow = np.array(f[name]["flow"]).copy()
        rho = np.array(f[name]["rho"]).copy()
        div = np.array(f[name]["div"]).copy()
        blob = np.array(f[name]["blob"]).copy()
    return img, mask, position, speed, diff, flow, rho, div, blob


def record(filename, img, mask, position, speed, diff, flow, rho, div, blob):
    """Record the results in a dataframe"""

    count, area, asymmetry_area, intensity, asymmetry_int = dna_blob_metrics(
        blob, img[:, 1]
    )
    df = pd.DataFrame(
        {
            "filename": [filename] * (position.shape[0] - 1),
            "frame": np.arange(position.shape[0] - 1),
            "position-x": position[:-1, 1],
            "position-y": position[:-1, 0],
            "displacement-x": speed[:, 1],
            "displacement-y": speed[:, 0],
            "dna mean intensity": average(np.expand_dims(img[:-1, 1], 1), mask[:-1]),
            "diff": average(diff, mask[:-1]),
            "flow": average(flow, mask[:-1]),
            "momemtum": average(rho, mask[:-1]),
            "divergence": average(div, mask[:-1]),
            "dna blob count": count[:-1],
            "dna blob area": area[:-1],
            "dna blob area asymmetry": asymmetry_area[:-1],
            "dna blob intensity": intensity[:-1],
            "dna blob intensity asymmetry": asymmetry_int[:-1],
        }
    )
    return df


def figure(
    ax, name, img, mask, position, speed, diff, flow, rho, div, blob, title=True
):
    """Create a figure with graphs over time"""

    count, area, asym_area, intensity, asym_int = dna_blob_metrics(blob, img[:, 1])

    ax[0].imshow(uv2rgb(img[0]))
    ax[0].set_axis_off()
    for c in measure.find_contours(mask[0, 0].astype(int), 0.5):
        ax[0].plot(c[:, 1], c[:, 0], "w")
    ax[0].plot(position[:, 1], position[:, 0], "w", linewidth=1)
    ax[0].text(-10, img.shape[2], Path(name).stem, fontsize=6, rotation=90)

    ax[1].plot(average(np.expand_dims(img[:-1, 1], 1), mask[:-1]))
    if title:
        ax[1].set(
            box_aspect=1, xlabel="time [frame]", title="dna mean intensity", ylim=0
        )
    else:
        ax[1].set(
            box_aspect=1, xlabel="time [frame]", ylabel="dna mean intensity", ylim=0
        )

    ax[2].plot(average(diff, mask[:-1]))
    if title:
        ax[2].set(box_aspect=1, xlabel="time [frame]", title="diff", ylim=0)
    else:
        ax[2].set(box_aspect=1, xlabel="time [frame]", ylabel="diff", ylim=0)

    ax[3].plot(average(flow, mask[:-1]))
    if title:
        ax[3].set(box_aspect=1, xlabel="time [frame]", ylabel="flow", ylim=0)
    else:
        ax[3].set(box_aspect=1, xlabel="time [frame]", title="flow", ylim=0)

    ax[4].plot(average(rho, mask[:-1]))
    if title:
        ax[4].set(box_aspect=1, xlabel="time [frame]", ylabel="momentum", ylim=0)
    else:
        ax[4].set(box_aspect=1, xlabel="time [frame]", title="momentum", ylim=0)

    ax[5].plot(average(div, mask[:-1]))
    if title:
        ax[5].set(box_aspect=1, xlabel="time [frame]", title="divergence", ylim=0)
    else:
        ax[5].set(box_aspect=1, xlabel="time [frame]", ylabel="divergence", ylim=0)

    ax[6].plot(count[:-1])
    if title:
        ax[6].set(box_aspect=1, xlabel="time [frame]", title="count", ylim=0)
    else:
        ax[6].set(box_aspect=1, xlabel="time [frame]", ylabel="count", ylim=0)

    ax[7].plot(area[:-1])
    if title:
        ax[7].set(box_aspect=1, xlabel="time [frame]", title="dna blob area", ylim=0)
    else:
        ax[7].set(box_aspect=1, xlabel="time [frame]", ylabel="dna blob area", ylim=0)

    ax[8].plot(asym_area[:-1])
    if title:
        ax[8].set(
            box_aspect=1, xlabel="time [frame]", title="dna blob area asymmetry", ylim=0
        )
    else:
        ax[8].set(
            box_aspect=1,
            xlabel="time [frame]",
            ylabel="dna blob area asymmetry",
            ylim=0,
        )
    ax[9].plot(intensity[:-1])
    if title:
        ax[9].set(
            box_aspect=1, xlabel="time [frame]", title="dna blob sum intenity", ylim=0
        )
    else:
        ax[9].set(
            box_aspect=1, xlabel="time [frame]", ylabel="dna blob sum intenity", ylim=0
        )

    ax[10].plot(asym_int[:-1])
    if title:
        ax[10].set(
            box_aspect=1,
            xlabel="time [frame]",
            title="dna blob intenity asymmetry",
            ylim=0,
        )
    else:
        ax[10].set(
            box_aspect=1,
            xlabel="time [frame]",
            ylabel="dna blob intenity asymmetry",
            ylim=0,
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
    step=5,
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
    indices = np.arange(0, img.shape[0], step)
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
