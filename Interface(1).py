import cv2
import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from tensorflow.keras.models import load_model
class DrawingApp:
    def __init__(self, root, n_im, reseau_model):
        self.root = root
        self.root.title("Code LaTeX à partir d'écriture manuscrite")
        self.n_im = n_im
        self.mainframe = tk.Frame(self.root)
        self.reseau_model = reseau_model
        self.create_canvas()
        self.class_name = ['(', ')', '+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'A', 'C', 'G',
                           'X', 'b', 'd', 'e', 'f', 'i', 'k', 'pi', 'times', 'u', 'v', 'w', 'y', 'z']


    def reset_interface(self):
        for widget in self.root.winfo_children():
            widget.destroy()
    def create_canvas(self):
        self.reset_interface()
        self.canvases = []
        self.images = []

        for i in range(self.n_im):
            canvas = tk.Canvas(self.root, bg="white", width=180, height=180)
            canvas.grid(row=0, column=i, sticky=tk.W + tk.N)
            canvas.bind("<B1-Motion>", lambda event, idx=i: self.paint(event, idx))

            image = Image.new("L", (180, 180), color="white")
            draw = ImageDraw.Draw(image)

            self.canvases.append(canvas)
            self.images.append((image, draw))


        # Boutons interface de base
        clear_button = tk.Button(root, text="Effacer", command=self.clear_canvas, font=("Helvetica", 16))
        valider_button = tk.Button(root, text="Valider", command=self.valider, font=("Helvetica", 16))

        valider_button.grid(column=12, row=0, sticky=tk.W)
        clear_button.grid(column=12, row=1, sticky=tk.W)


    def paint(self, event, idx):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y - 1)
        self.canvases[idx].create_oval(x1, y1, x2, y2, fill="black", width=2)
        self.images[idx][1].line([x1, y1, x2, y2], fill="black", width=2)

    def clear_canvas(self):
        for canvas in self.canvases:
            canvas.delete("all")

        self.images = [(Image.new("L", (180, 180), color="white"),
                        ImageDraw.Draw(Image.new("L", (180, 180), color="white")))
                       for _ in range(self.n_im)]


    def monreseauestropleplusmeilleur(self, image):

        #On resize l'image à la taille des entrées de notre reseau
        resized_image = image.resize((45, 45), Image.LANCZOS)
        img_filtered = resized_image.filter(ImageFilter.SHARPEN)
        img_binary = resized_image.point(lambda x: 0 if x < 170 else 255)

        image = np.array(img_binary)

        # On applique la prédiction avec le modèle
        predictions = self.reseau_model.predict(np.expand_dims(image, axis=0))

        # On utilise argmax pour obtenir la classe prédite
        predicted_class = np.argmax(predictions)

        # Le nom de la classe
        predicted_class_name = self.class_name[predicted_class]
        return predicted_class_name

    def copier_contenu(self, widget):
        contenu = widget.cget("text")  # Récupérer le contenu du widget Label
        self.root.clipboard_clear()  # Effacer le presse-papiers
        self.root.clipboard_append(contenu)  # Copier le contenu dans le presse-papiers
        self.root.update()  # Mettre à jour la fenêtre principale
    def valider(self):

        self.reset_interface()

        retour_button = tk.Button(self.root, text="Retour", command=lambda: self.create_canvas(), font=("Helvetica", 16))
        caractere = [self.monreseauestropleplusmeilleur(image) for image,_ in self.images]  # Donne la classe retournee par le reseau pour chaque image)
        affichage = ""

        for i in range(self.n_im):
            affichage += str(caractere[i])
            print(caractere[i])
        text_detat = tk.Label(self.root, text=f"Votre équation :", font=("Helvetica", 16))
        resultat_label = tk.Label(self.root, text=affichage, font=("Helvetica", 69))

        retour_button.grid(column=self.n_im // 2, row=4, sticky=tk.W)
        text_detat.grid(column=self.n_im // 2 - 1, row=5, sticky=tk.W)
        resultat_label.grid(column=self.n_im // 2, row=5, sticky=tk.W)


        # Créer un bouton pour copier le contenu
        bouton_copier = tk.Button(root, text="Copier", command=self.copier_contenu(resultat_label), font=("Helvetica", 16))
        bouton_copier.grid(column=self.n_im //2, row=6, sticky=tk.W)


if __name__ == "__main__":
    trained_model = load_model('C:/Users/arthu/OneDrive/Documents/Albert_pond.keras')
    root = tk.Tk()
    app = DrawingApp(root, 5, trained_model)
    root.mainloop()















