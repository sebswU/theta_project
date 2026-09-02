#convert camera output (mainly skellycam websocket output) to NCHW format for model inference
import numpy as np

def nchw(frame: np.ndarray, H, W, C, B) -> np.ndarray:
    """
    Convert a frame from HWC format to NCHW format.

    Args:
        frame (np.ndarray): The input frame in HWC format.
        H (int): Height of the frame.
        W (int): Width of the frame.
        C (int): Number of channels in the frame.

    Returns:
        np.ndarray: The frame in NCHW format.
    """
    # Assuming the input frame is in HWC format
    # Create an empty array for NCHW format
    nchw_frame = np.zeros((1, C, H, W), dtype=np.float32)

    # Rearrange the data from HWC to NCHW
    for c in range(C):
        nchw_frame[0, c, :, :] = frame[:, :, c]

    return nchw_frame

def convert_frames_to_nchw(frames: list[np.ndarray]) -> list[np.ndarray]:
    """
    Convert a list of frames from HWC format to NCHW format.

    Args:
        frames (list[np.ndarray]): A list of input frames in HWC format.

    Returns:
        list[np.ndarray]: A list of frames in NCHW format.
    """
    nchw_frames = []
    for frame in frames:
        H, W, C = frame.shape
        nchw_frame = nchw(frame, H, W, C, 1)
        nchw_frames.append(nchw_frame)
    
    return nchw_frames