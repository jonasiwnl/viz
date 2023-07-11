from image_to_pc import create_point_cloud


def main():
    image_path = "static/2.jpg"

    create_point_cloud(image_path)

    # surface reconstruction

    # maybe: mesh simplification/processing

    # finally, visualization


if __name__ == "__main__":
    main()
