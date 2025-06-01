import cv2
import random
from pathlib import Path
from tqdm import tqdm
import hashlib

def unique_name(data):
    """
    Calculate the MD5 hash of the given data.
    Args:
        data (bytes): Data to hash.
    Returns:
        str: MD5 hash of the data.
    """
    md5_hash = hashlib.md5()
    md5_hash.update(data)
    return md5_hash.hexdigest()

def sample_frames(video_path, N, seed=42):
    """
    Sample N random frames from a video file.
    Args:
        video_path (str or Path): Path to the video file.
        N (int): Number of frames to sample.
        seed (int): Random seed for reproducibility.
    Returns:
        list: List of sampled frames.
        list: List of indices of the sampled frames.
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if N > total_frames:
        raise ValueError(f"N ({N}) is greater than total number of frames ({total_frames})")

    # Set seed for reproducibility
    random.seed(seed)
    selected_indices = sorted(random.sample(range(total_frames), N))

    frames = []
    for idx in tqdm(selected_indices, desc=f"Sampling frames from {video_path.name}", unit="frame"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Unable to read frame at index {idx}")
            continue
        frames.append(frame)

    cap.release()
    return frames, selected_indices

def save_frame(frame, output_folder):
    """
    Save a single frame to the specified output folder.
    Args:
        frame (numpy.ndarray): The frame to save.
        output_folder (Path): The folder where the frame will be saved.
    """
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)

    frame_name = unique_name(frame.tobytes())
    output_path = output_folder / f"{frame_name}.jpg"
    cv2.imwrite(str(output_path), frame)

    return output_path


FOOTAGE_DIR = Path("./footage")
SAMPLE_DIR = Path("./samples")
SAMPLES_PER_VIDEO = 100
SEED = 42

if __name__ == "__main__":
    extensions = [".MP4", ".mp4"]

    # Select video files with specified extensions
    video_files = list(FOOTAGE_DIR.glob("*.*"))

    # Filter video files by extensions
    video_files = [f for f in video_files if f.suffix in extensions]

    if not video_files:
        raise FileNotFoundError("No video files found in the specified directory.")

    # Sample frames from each video file
    for video_file in video_files:
        try:
            frames, indices = sample_frames(video_file, N=SAMPLES_PER_VIDEO, seed=SEED)
        except Exception as e:
            print(f"Error processing {video_file}: {e}")
            continue

        print(f"Sampled {len(frames)} frames from {video_file} at indices {indices}")

        # Save the sampled frames to the output directory
        for i, frame in tqdm(enumerate(frames), desc=f"Saving frames from {video_file.name}", unit="frame"):
            output_folder = SAMPLE_DIR / video_file.stem
            output_path = save_frame(frame, output_folder)
            # print(f"Saved frame {i+1}/{len(frames)} to {output_path}")
        print(f"Completed sampling from {video_file}\n")
    print("All videos processed.")