from PIL import Image  # image manipulation


def make_grid(images, rows, cols):
    w, h = images[0].size                        # assumes all same size

    grid = Image.new("RGB", (cols * w, rows * h)) # blank canvas

    for i, img in enumerate(images):
        col = i % cols                            # column position
        row = i // cols                           # row position
        grid.paste(img, (col * w, row * h))       # place image in slot

    return grid
