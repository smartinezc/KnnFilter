
import numpy as np
import matplotlib.pyplot as plt
import io
from collections import Counter


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

            if not self.data[clase].any():
                self.data[clase] = np.array(datLinea);
            else:
                self.data[clase] = np.vstack([self.data[clase], datLinea]);

        archivoTexto.close();

    def darDatosEntrada(self, clase=-1):
        if clase == -1:
            return self.data;
        else:
            return self.data[clase];

    def suavizarDatosKNN(self, atributos, k=-1):
        #Si no se especifica un k, se emplea el número de clases más uno, así siempre habrá una clase con mayoría
        if k == -1:
            k = self.numClases+1

        #Construye la matriz de datos suavizados, en principio es igual a los datos de entrada
        #datSuavizado = self.data

        #Itera sobre las clases de los datos y almacena en d cada punto
        d = []
        for cl,dat in enumerate(self.data):
            numF = dat.shape[0]

            #Itera sobre los datos de la clase actual según los atributos especificados
            for f in range(numF):
                punto = dat[f, atributos]
                #d es una lista de 3 columnas: las coordenadas de los atributos y la clase
                d.append([punto[0], punto[1], cl])
        dS = d
        print(len(dS))

        #Itera sobre los puntos
        for inD,pD in enumerate(d):
            #Calcular las distancias del punto pD con los demás datos, y las almacena en dist
            dist = []
            for n in range(len(d)):
                if n == inD:
                    dist.append(444444)
                else:
                    dist.append(np.sqrt((pD[0] - d[n][0])**2 + (pD[1] - d[n][1])**2))

            #Obtener las k distancias más cercanas y las clases de estos
            kPuntosMasCerca = np.argsort(dist)[:k]
            kClasesMasCerca = [d[i][2] for i in kPuntosMasCerca]

            #Clase con más puntos cercanos
            claseMasCerca = Counter(kClasesMasCerca).most_common(1)[0][0]

            #Si la clase con más puntos cercanos no es la clase del punto pD, este se borra
            if pD[2] != claseMasCerca:
                dS.pop(inD)

        print(len(dS))
        return dS;


    def graficarDatosEntrada(self, atributos, ruta):
        plt.title("Dispersión de datos de entrada")
        plt.xlabel("Atributo {}".format(atributos[0]))
        plt.ylabel("Atributo {}".format(atributos[1]))
        for cl,dat in enumerate(self.data):
            plt.scatter(dat[:,atributos[0]], dat[:,atributos[1]], label = 'Clase {}'.format(cl));

        plt.legend(loc=0);
        plt.savefig(ruta);
        plt.close();

    def graficarDatos(self, datos, atributos, ruta):
        plt.title("Dispersión de datos suavizados")
        plt.xlabel("Atributo {}".format(atributos[0]))
        plt.ylabel("Atributo {}".format(atributos[1]))

        datosForma = [np.zeros(len(atributos)) for x in range(self.numClases)]
        for dato in datos:
            clase = dato[2]
            datLinea = [dato[0], dato[1]]
            if not datosForma[clase].any():
                datosForma[clase] = np.array(datLinea)
            else:
                datosForma[clase] = np.vstack([datosForma[clase], datLinea])

        for cl,dat in enumerate(datosForma):
            plt.scatter(dat[:,0], dat[:,1], label = 'Clase {}'.format(cl));

        plt.legend(loc=3);
        plt.savefig(ruta);
        plt.close();


    def estado(self):
        print(self.data);


app = suavizado("seg-data.txt")
app.graficarDatosEntrada([0, 1], "prueba.jpg")
app.graficarDatos(app.suavizarDatosKNN([0, 1]), [0, 1], "pruebaSuav.jpg")
