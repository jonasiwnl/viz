from multiple_pc import create_point_clouds_sameangle


def main():
    image_paths = [
        "static/sameangle/1.jpg",
        "static/sameangle/2.jpg",
    ]

    create_point_clouds_sameangle(image_paths)

    # surface reconstruction

    # maybe: mesh simplification/processing

    # finally, visualization


if __name__ == "__main__":
    main()
