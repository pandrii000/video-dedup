import torch
import torchvision
import cv2
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def get_duplicates(videos_dir: Path) -> list[tuple[Path, Path, float]]:
    """
    This is a function, that takes a path to a folder with videos
    and returns a list of duplicates accross these videos
    a duplicate is a video, that is a near-copy of another video
    which means it could be resized, slightly changed the color, etc.
    the function should return a list of tuples (video1, video2, similarity)
    where video1 and video2 are paths to videos and similarity is a float
    between 0 and 1, where 0 means no similarity and 1 means the videos are identical
    the list should be sorted by similarity in descending order
    the function should be able to handle videos of different resolutions
    and different framerates and lengths.
    It uses the following approach: calculate the mean feature vector for each frame in a video,
    then average them to get a single vector for the whole video.
    Then calculate the cosine similarity between the vectors of two videos.
    For the feature extraction use MobileNet v3 small pretrained on imagenet.
    Before calculating the similarity, resize the videos to 224x224.
    """

    # Load the model
    mobilenetv3 = torchvision.models.mobilenet_v3_small(pretrained=True)
    model = torch.nn.Sequential(*list(mobilenetv3.children())[:-1])
    model.eval()

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
    ])

    videos = sorted(videos_dir.iterdir())
    videos_features = []
    
    # Iterate over all videos in the directory
    for video_path in tqdm(videos):
        # Load the video
        cap = cv2.VideoCapture(str(video_path))
        # Get the number of frames
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create a list of feature vectors for each frame
        features = []
        for i in range(n_frames):
            ret, frame = cap.read()
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame = transform(frame)
            frame = frame.unsqueeze(0)
            with torch.no_grad():
                feature = model(frame).flatten()
            features.append(feature)
        
        # Calculate the mean feature vector for the whole video
        mean_feature = torch.mean(torch.stack(features), dim=0)
        # Add the video and its feature vector to the list
        videos_features.append((video_path, mean_feature))
    
    results = []
        
    # Calculate the similarity between all pairs of videos
    for i in range(len(videos_features)):
        for j in range(i + 1, len(videos_features)):
            # Calculate the cosine similarity
            similarity = torch.cosine_similarity(videos_features[i][1], videos_features[j][1], dim=0)
            # Add the pair of videos and their similarity to the list
            results.append((videos_features[i][0], videos_features[j][0], similarity.item()))
            
    # Sort the list by similarity in descending order
    results.sort(key=lambda x: x[2], reverse=True)
    
    return results
    

if __name__ == "__main__":
    videos_dir = Path("videos")
    duplicates = get_duplicates(videos_dir)

    # print the results
    for video1, video2, similarity in duplicates:
        print(f"{video1} {video2} {similarity}")