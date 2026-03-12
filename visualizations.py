#Helper functions
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import make_lupton_rgb
import torch

from scipy.ndimage import gaussian_filter

class Visualization:

    RING_CLASS_MAPPING = {
        0: "0 - No Ring",
        1: "1 - Inner Ring Only",
        2: "2 - Outer Ring Only",
        3: "3 - Both Rings"
    }

    @staticmethod
    def plot_rgb_image(df, transform=None):
        """ Plotea cinco ejemplos por clase.

        Args:
            df (pd.DataFrame): DataFrame que contiene las rutas de los archivos FITS y las clases de anillo.
            transform (callable, optional): Función de transformación a aplicar a las imágenes. Defaults to None.
        """
        # Se utiliza un grid de 4 filas y 5 columnas para mostrar 5 ejemplos por cada una de las 4 clases.
        fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(20, 16))
        for i, sample in enumerate(df.to_dict(orient='records')):
            with fits.open(sample['file_loc']) as hdul:
                data = hdul[0].data
                data = np.asarray(data, dtype=np.float32)
                
                image_tensor = torch.from_numpy(data)
                if transform:
                    rgb_image = transform(image_tensor)
                    # Convert Tensor to NumPy array if needed
                    if isinstance(rgb_image, torch.Tensor):
                        rgb_image = rgb_image.detach().cpu().numpy()
                else:
                    #apply default lupton rgb transform
                    rgb_image = Transformations.channels_to_rgb(data[1], data[0], data[2])
                row = i // 5
                col = i % 5
                axes[row, col].imshow(rgb_image)
                axes[row, col].set_title(f"Ring Class: {Visualization.RING_CLASS_MAPPING[sample['ring_class']]}\n ID: {sample['id_str']}", fontsize=10)
                axes[row, col].axis('off')

        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def get_raw_fits(df, id_str):
        """ Extrae la imagen FITS sin procesar por ID específico.

        Args:
            df (pd.DataFrame): DataFrame que contiene las rutas de los archivos FITS y las clases de anillo.
            id_str (str): ID del objeto a extraer.

        Returns:
            torch.Tensor: Tensor con la imagen FITS sin procesar.
        """
        filtered = df[df['id_str'] == id_str]
        if filtered.empty:
            raise ValueError(f"No se encontró ningún ejemplo con id_str='{id_str}'.")
        sample = filtered.iloc[0]
        with fits.open(sample['file_loc']) as hdul:
            data = hdul[0].data
            data = np.asarray(data, dtype=np.float32)
            image_tensor = torch.from_numpy(data)

            return image_tensor

    @staticmethod
    def extract_example(df, id_str, plot_rgb=False, transform=None):
        """ Extrae un ejemplo por ID específico.

        Args:
            df (pd.DataFrame): DataFrame que contiene las rutas de los archivos FITS y las clases de anillo.
            id_str (str): ID del objeto a extraer.
            plot_rgb (bool, optional): Si es True, se muestra la imagen RGB. Si transform no es None, se muestra la imagen RGB y la transformada juntas. Defaults to False.
            transform (callable, optional): Función de transformación a aplicar a la imagen. Defaults to None.

        Returns:
            np.ndarray: Imagen RGB del ejemplo extraído.
        """
        filtered = df[df['id_str'] == id_str]
        if filtered.empty:
            raise ValueError(f"No se encontró ningún ejemplo con id_str='{id_str}'.")
        sample = filtered.iloc[0]
        with fits.open(sample['file_loc']) as hdul:
            data = hdul[0].data
            data = np.asarray(data, dtype=np.float32)
            image_tensor = torch.from_numpy(data)

            # Always build the default RGB view for consistent visualization.
            rgb_image = Transformations.channels_to_rgb(data[1], data[0], data[2])
            transformed_image = transform(image_tensor) if transform else None
            
            if plot_rgb:
                base_title = f"Ring Class: {Visualization.RING_CLASS_MAPPING[sample['ring_class']]}\n ID: {sample['id_str']}"

                if transformed_image is not None:
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                    axes[0].imshow(rgb_image)
                    axes[0].set_title(f"RGB Original\n{base_title}", fontsize=10)
                    axes[0].axis('off')

                    axes[1].imshow(transformed_image)
                    axes[1].set_title("Imagen Transformada", fontsize=10)
                    axes[1].axis('off')

                    plt.tight_layout()
                    plt.show()
                else:
                    plt.imshow(rgb_image)
                    plt.title(base_title, fontsize=10)
                    plt.axis('off')
                    plt.show()
            
            return transformed_image if transformed_image is not None else rgb_image, 
        
    @staticmethod
    def plot_3d_channels(image, transform=None):
        """ Plotea las tres bandas de una imagen en 3D.

        Args:
            image (torch.Tensor): Tensor de la imagen con forma (C, H, W) o (H, W, C).
            transform (callable, optional): Función de transformación a aplicar a la imagen. Defaults to None.
        """
        if transform:
            transformed_image = transform(image)
            if isinstance(transformed_image, torch.Tensor):
                image_np = transformed_image.detach().cpu().numpy()
            else:
                image_np = np.asarray(transformed_image)
        else:
            # Si no se proporciona una transformación, se asume que la imagen ya está en formato NumPy o Tensor.
            image_np = np.asarray(image, dtype=np.float32)
        
        #plot 3d image (height, width, pixel intensity) 
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(np.arange(image_np.shape[2]), np.arange(image_np.shape[1]))
        for i in range(image_np.shape[0]):
            ax.plot_surface(X, Y, image_np[i, :, :], cmap='viridis', alpha=0.5)
        ax.set_title("3D Visualization of Image Channels")
        ax.set_xlabel("Width")
        ax.set_ylabel("Height")
        ax.set_zlabel("Pixel Intensity")
        plt.show()

    @staticmethod
    def plot_rgb_from_tensor(image, transform=None):
        """ Plotea la imagen RGB a partir de un tensor.

        Args:
            image (torch.Tensor): Tensor de la imagen con forma (C, H, W) o (H, W, C).
            transform (callable, optional): Función de transformación a aplicar a la imagen. Defaults to None.
        """
        if transform:
            transformed_image = transform(image)
            if isinstance(transformed_image, torch.Tensor):
                image_np = transformed_image.detach().cpu().numpy()
            else:
                image_np = np.asarray(transformed_image)
        else:
            image_np = np.asarray(image, dtype=np.float32)

        # Si la imagen transformada tiene 3 canales, se asume que ya es RGB y se muestra directamente.
        if image_np.shape[0] == 3:
            plt.imshow(np.transpose(image_np, (1, 2, 0)))  # Convertir de (C, H, W) a (H, W, C)
            plt.title("Imagen RGB Transformada", fontsize=10)
            plt.axis('off')
            plt.show()
            return torch.from_numpy(image_np)
            


    @staticmethod
    def plot_3d_tensor(image, transform=None):
        """ Plotea las tres bandas de una imagen en 3D.

        Args:
            image (torch.Tensor): Tensor de la imagen con forma (C, H, W) o (H, W, C).
            transform (callable, optional): Función de transformación a aplicar a la imagen. Defaults to None.
        """
        if transform:
            transformed_image = transform(image)
            if isinstance(transformed_image, torch.Tensor):
                image_np = transformed_image.detach().cpu().numpy()
            else:
                image_np = np.asarray(transformed_image)
        else:
            # Si no se proporciona una transformación, se asume que la imagen ya está en formato NumPy o Tensor.
            image_np = np.asarray(image, dtype=np.float32)
        
        #plot 3d image (height, width, pixel intensity) 
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(np.arange(image_np.shape[2]), np.arange(image_np.shape[1]))
        for i in range(image_np.shape[0]):
            ax.plot_surface(X, Y, image_np[i, :, :], cmap='viridis', alpha=0.5)
        ax.set_title("3D Visualization of Image Channels")
        ax.set_xlabel("Width")
        ax.set_ylabel("Height")
        ax.set_zlabel("Pixel Intensity")
        plt.show()

    @staticmethod
    def plot_raw_fits(image, transform=None):
        """ Plotea la imagen FITS sin procesar en 3D.

        Args:
            image (torch.Tensor): Tensor de la imagen con forma (C, H, W) o (H, W, C).
            transform (callable, optional): Función de transformación a aplicar a la imagen. Defaults to None.
        """
        if transform:
            transformed_image = transform(image)
            if isinstance(transformed_image, torch.Tensor):
                image_np = transformed_image.detach().cpu().numpy()
            else:
                image_np = np.asarray(transformed_image)
        else:
            # Si no se proporciona una transformación, se asume que la imagen ya está en formato NumPy o Tensor.
            image_np = np.asarray(image, dtype=np.float32)
        #plot 3d image (height, width, pixel intensity)
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(np.arange(image_np.shape[2]), np.arange(image_np.shape[1]))
        for i in range(image_np.shape[0]):
            ax.plot_surface(X, Y, image_np[i, :, :], cmap='viridis', alpha=0.5)
        ax.set_title("3D Visualization of Raw FITS Image")
        ax.set_xlabel("Width")
        ax.set_ylabel("Height")
        ax.set_zlabel("Pixel Intensity")
        plt.show()

    @staticmethod
    def print_tensor_channel_ranges(tensor):
        """ Imprime rangos de valores por canal de un tensor dado

        Args:
            tensor (torch.Tensor): Tensor de la imagen con forma (C, H, W)
        """
        if isinstance(tensor, torch.Tensor):
            image_np = tensor.detach().cpu().numpy()
        else:
            image_np = np.asarray(tensor)

        r = pd.DataFrame(image_np[0])  # Convert the first channel to a DataFrame for analysis
        g = pd.DataFrame(image_np[1])  # Convert the second channel to a DataFrame for analysis
        z = pd.DataFrame(image_np[2])  # Convert the third channel to a DataFrame for analysis

        #Get numerical ranges of the entire dataframe R, G, Z
        print("R channel range:", r.min().min(), "to", r.max().max())
        print("G channel range:", g.min().min(), "to", g.max().max())
        print("Z channel range:", z.min().min(), "to", z.max().max())


class Transformations:
    @staticmethod
    def channels_to_rgb(r, g, z, stretch=0.5, Q=10):
        """ Convertir los canales en una imágen RGB compuesta.

        Args:
            r (np.ndarray): Canal rojo.
            g (np.ndarray): Canal verde.
            z (np.ndarray): Canal azul.

        Returns:
            np.ndarray: Imagen RGB compuesta.
        """
        # Se utiliza la función make_lupton_rgb debido a que ayuda
        # a manejar el alto rango de imágenes astronómicas controlando
        # el contraste y la saturación, manteniendo el color.
        # Convert input channels to float to avoid UFuncTypeError
        r_float = r.astype(np.float32)
        g_float = g.astype(np.float32)
        z_float = z.astype(np.float32)
        rgb_img = make_lupton_rgb(r_float, g_float, z_float, stretch=stretch, Q=Q)
        return rgb_img
    
    @staticmethod
    def unsharp_mask(image, sigma=1.0, amount=1.0, threshold=0):
        """
        Applies unsharp masking to a numpy array.
        
        Parameters:
        - image: input numpy array (2D for grayscale, 3D for RGB).
        - sigma: standard deviation for Gaussian blur (controls the width of the edges).
        - amount: multiplier for the sharpening effect (usually 0.5 to 2.0).
        - threshold: minimum brightness change to be sharpened (helps reduce noise).
        """
        # 1. Create the blurred version of the image
        # For RGB images, we don't blur along the color channel axis
        if image.ndim == 3:
            blurred = gaussian_filter(image, sigma=(sigma, sigma, 0))
        else:
            blurred = gaussian_filter(image, sigma=sigma)
        
        # 2. Calculate the mask (Original / Blurred)
        mask = image - blurred
        
        # 3. Apply thresholding if needed to avoid sharpening noise
        if threshold > 0:
            mask[np.abs(mask) < threshold] = 0
        
        # 4. Add the weighted mask back to the original image
        sharpened = image + amount * mask
        
        return blurred, sharpened