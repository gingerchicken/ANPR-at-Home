from ultralytics import YOLO

model = YOLO('last.pt')

# Run inference on a video file
results = model.predict(
    source=input("Test Path: "),  # path to your video
    save=True,                    # save annotated video to disk
    conf=0.4,                     # confidence threshold
    stream=False                  # set to True to show live in a window
)