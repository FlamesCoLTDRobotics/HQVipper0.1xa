
import streamlit as st

import numpy as np

from PIL import Image

def main():

    # @st.cache

    def inpaint(img, mask, iterations=1, radius=1):

        """Inpaints an image using the Fast Marching method.

        Args:

            img: An array of shape (h, w, 3) representing the image to be inpainted.

            mask: An array of shape (h, w) representing the inpainting mask.

            iterations: The number of iterations of the Fast Marching method to perform.

            radius: The radius of the neighborhood used in the Fast Marching method.

        Returns:

            An array of shape (h, w, 3) containing the inpainted image.

        """

        h, w, _ = img.shape

        d = np.zeros((h, w))

        d[mask] = 1

        for _ in range(iterations):

            d = fast_marching(d, radius)

        inpainted = img.copy()

        inpainted[mask] = 0

        inpainted[d == 0] = img[d == 0]

        return inpainted



def fast_marching(d, radius):

    """Performs a single iteration of the Fast Marching method.

    Args:

        d: An array of shape (h, w) containing the distance transform of the inpainting mask.

        radius: The radius of the neighborhood used in the Fast Marching method.

    Returns:

        An array of shape (h, w) containing the updated distance transform.

    """

    h, w = d.shape

    offsets = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])

    for i in range(h):

        for j in range(w):

            if d[i, j] > 0:

                continue

            for offset in offsets:

                x = i + offset[0]

                y = j + offset[1]

                if x < 0 or x >= h or y < 0 or y >= w:

                    continue

                dist = d[x, y] + np.linalg.norm(offset)

                if dist < radius:

                    d[i, j] = dist

    return d



if __name__ == '__main__':

    main()
