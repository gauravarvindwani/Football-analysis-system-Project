from utils import read_video, save_video
from trackers import Tracker
# import cv2
# import numpy as np
# from team_assigner import TeamAssigner
# from player_ball_assigner import PlayerBallAssigner
# from camera_movement_estimator import CameraMovementEstimator
# from view_transformer import ViewTransformer
# from speed_and_distance_estimator import SpeedAndDistance_Estimator


def main():
    # Read Video
    video_frames = read_video('input_videos/08fd33_4.mp4')

    #initialize tracker
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(video_frames,read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')

    # Draw output 
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # Save video
    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()