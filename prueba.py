
import numpy as np;
import io;


class suavizado():

    def __init__(self, rutaDatos):
        archivoTexto = io.open(rutaDatos, 'r');
        lineas = archivoTexto.readlines();

        self.numLineas = int(lineas[0]);
        self.numAtributos = int(lineas[1]);
        self.numClases = int(lineas[2]);

        self.data = [np.zeros(self.numAtributos) for x in range(self.numClases)];
        for l in range(3, self.numLineas+3):
            clase = int(lineas[l].split(',')[self.numAtributos]);
            datLinea = lineas[l].split(',')[0:self.numAtributos];
            datLinea = [float(i) for i in datLinea];
            self.data[clase] = np.vstack([self.data[clase], datLinea]);


        print(len(self.data[0]));
        archivoTexto.close();


    def estado(self):
        print('El archivo {} se inicializó con las siguientes características: \nNúmero Atributos: {} \nNúmero Clases {}'.format(self.ruta, self.numAtributos, self.numClases));


app = suavizado("seg-data.txt");
