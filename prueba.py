
import numpy as np;
import matplotlib.pyplot as plt;
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

    def darDatosSuavizados(self, clase=-1):
        datSuavizado = self.data;
        for cl,dat in enumerate(self.data):
            numF = dat.shape[0];
            numC = dat.shape[1];
            for c in range(numC):
                for f in range(numF):
                    cont, sum = 1, 0;
                    if f-1 > 0:
                        sum += dat[f-1, c];
                        cont += 1;

                    sum += dat[f, c];

                    if f+1 < numF:
                        sum += dat[f+1, c];
                        cont += 1;

                    dat[f, c] = sum/cont;

            datSuavizado[cl] = dat;
        if clase == -1:
            return datSuavizado;
        else:
            return datSuavizado[clase]


    def graficar(self, datos, atributos, ruta):
        plt.title("Dispersión de datos");
        for cl,dat in enumerate(datos):
            plt.scatter(dat[atributos[0]], dat[atributos[1]], label = 'Clase {}'.format(cl));

        plt.legend(loc=0);
        plt.savefig(ruta);
        plt.close();


    def estado(self):
        print(self.data);


app = suavizado("seg-data.txt");
app.graficar(app.darDatosEntrada(), [2, 3], "prueba.jpg");
app.graficar(app.darDatosSuavizados(), [2, 3], "pruebaSua.jpg");
