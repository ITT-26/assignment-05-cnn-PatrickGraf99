from argparse import ArgumentParser
from camera_app import CameraApp

def main():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=False, default='64',
                        choices=['16', '32', '64', '128', '256', '512'],
                        help="The image size that the CNN should use")
    parser.add_argument("-o", "--output", required=False, default='.', help="The output folder")
    parser.add_argument("-c", "--camera", required=False, type=int, default=0, help="Camera device ID")
    args = parser.parse_args()
    save_path = args.output
    model_size = args.model
    camera_id = args.camera
    camera_app = CameraApp(model_size, save_path, camera_id)
    camera_app.start()

if __name__ == '__main__':
    main()