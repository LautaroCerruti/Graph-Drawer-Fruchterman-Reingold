#! /usr/bin/python

# Complementos Matematicos I
# Cassinerio Marcos, Cerruti Lautaro

import argparse
from math import sqrt
from numpy import random as nprandom

import matplotlib.pyplot as plt


class LayoutGraph:

    def __init__(self, grafo, iters, temperature, refresh, c1, c2, width, height,
                 gravity, cooling, save, verbose=False):
        """
        Parámetros:
        grafo: grafo en formato lista
        iters: cantidad de iteraciones a realizar
        temperature: temperatura inicial del sistema
        refresh: cada cuántas iteraciones graficar. Si su valor es cero, entonces debe graficarse solo al final.
        c1: constante de repulsión
        c2: constante de atracción
        width: Tamaño del eje X
        height: Tamaño del eje Y
        gravity: constante de gravedad
        cooling: constante de enfriamiento
        save: archivo donde guardar
        verbose: si está encendido, activa los comentarios
        """

        # Guardo el grafo
        self.grafo = grafo

        # Inicializo estado
        # Completar
        self.posiciones = {}
        self.fuerzas = {}

        # Guardo opciones
        self.iters = iters
        self.verbose = verbose
        self.temperature = temperature
        self.refresh = refresh
        self.c1 = c1
        self.c2 = c2
        self.width = width
        self.height = height
        self.temperatureConstant = cooling
        self.g = gravity
        self.save = save

        area_ratio = sqrt(self.width * self.height / len(self.grafo[0]))
        self.kAttraction = c2 * area_ratio
        self.kRepulsion = c1 * area_ratio

        self.error = 1

        plt.rcParams["figure.figsize"] = (self.width / 100 + 3, self.height / 100 + 3)

    def layout(self):
        """
            Aplica el algoritmo de Fruchtermann-Reingold para obtener (y mostrar)
            un layout
        """
        self.randomize_positions()
        self.print_message("Temperatura inicial: " + str(self.temperature))
        for i in range(self.iters):
            self.fruchterman_reingold_step()
            if self.refresh != 0 and i % self.refresh == 0:
                self.show_graph()
        if self.save:
            plt.savefig(self.save)
        self.show_graph()
        plt.show()

    def print_message(self, m):
        if self.verbose:
            print(m)

    def show_graph(self):
        self.print_message("Imprimiendo Grafo")
        plt.clf()
        plt.xlim([0, self.width])
        plt.ylim([0, self.height])
        x = [self.posiciones[v][0] for v in self.grafo[0]]
        y = [self.posiciones[v][1] for v in self.grafo[0]]
        plt.scatter(x, y, color='blue')
        for v1, v2 in self.grafo[1]:
            plt.plot([self.posiciones[v1][0], self.posiciones[v2][0]],
                     [self.posiciones[v1][1], self.posiciones[v2][1]],
                     color='green')
        plt.pause(0.001)

    def randomize_positions(self):
        self.print_message("Colocando Posiciones Aleatorias")
        for vertice in self.grafo[0]:
            self.posiciones[vertice] = (nprandom.uniform(0, self.width), nprandom.uniform(0, self.height))

    def initialize_forces(self):
        self.print_message("Inicializando Fuerzas")
        for v in self.grafo[0]:
            self.fuerzas[v] = (0.0, 0.0)

    def compute_attraction_forces(self):
        self.print_message("Computando Fuerzas de Atraccion")
        for v1, v2 in self.grafo[1]:
            distance = self.calculate_distance(self.posiciones[v1], self.posiciones[v2])
            if distance > self.error:
                mod_fa = self.f_attraction(distance)
                force = self.calculate_force(mod_fa, distance, self.posiciones[v1], self.posiciones[v2])
                self.print_message("Fuerza de atraccion de " + v1 + " a " + v2 + ": " + str(force))
                self.fuerzas[v1] = self.fuerzas[v1][0] + force[0], self.fuerzas[v1][1] + force[1]
                self.fuerzas[v2] = self.fuerzas[v2][0] - force[0], self.fuerzas[v2][1] - force[1]

    def compute_repulsion_forces(self):
        self.print_message("Computando Fuerzas de Respulsion")
        for v1 in self.grafo[0]:
            for v2 in self.grafo[0]:
                if v1 != v2:
                    distance = self.calculate_distance(self.posiciones[v1], self.posiciones[v2])
                    if distance > self.error:
                        mod_fr = self.f_respulsion(distance)
                        force = self.calculate_force(mod_fr, distance, self.posiciones[v1], self.posiciones[v2])
                    else:
                        force = nprandom.rand(2) * 10
                    self.print_message("Fuerza de repulsion de " + v1 + " a " + v2 + ": " + str(force))
                    self.fuerzas[v1] = self.fuerzas[v1][0] - force[0], self.fuerzas[v1][1] - force[1]
                    self.fuerzas[v2] = self.fuerzas[v2][0] + force[0], self.fuerzas[v2][1] + force[1]

    @staticmethod
    def calculate_distance(v1, v2):
        return sqrt((v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2)

    @staticmethod
    def calculate_force(mod_fa, distance, v1, v2):
        return ((mod_fa * (v2[0] - v1[0])) / distance,
                (mod_fa * (v2[1] - v1[1])) / distance)

    def update_positions(self):
        for v in self.grafo[0]:
            modulo = sqrt(self.fuerzas[v][0] ** 2 + self.fuerzas[v][1] ** 2)
            if modulo > self.temperature:
                self.fuerzas[v] = (self.fuerzas[v][0] * self.temperature / modulo,
                                   self.fuerzas[v][1] * self.temperature / modulo)
            self.print_message("Actualizando la posicion de " + str(v) + " con una fuerza " + str(self.fuerzas[v]))
            pos = self.posiciones[v][0] + self.fuerzas[v][0], self.posiciones[v][1] + self.fuerzas[v][1]
            if pos[0] < 0.0:
                pos = 0.0, pos[1]
            if pos[0] > float(self.width):
                pos = float(self.width), pos[1]
            if pos[1] < 0.0:
                pos = pos[0], 0.0
            if pos[1] > float(self.height):
                pos = pos[0], float(self.height)
            self.posiciones[v] = pos

    def f_attraction(self, distance):
        return (distance ** 2) / self.kAttraction

    def f_respulsion(self, distance):
        return (self.kRepulsion ** 2) / distance

    def compute_gravity_forces(self):
        self.print_message("Computando Fuerza de Gravedad")
        for v in self.grafo[0]:
            center = self.width / 2, self.height / 2
            distance = self.calculate_distance(self.posiciones[v], center)
            if distance > self.error:
                force = self.calculate_force(self.g, distance, self.posiciones[v], center)
                force = force[0] / distance, force[1] / distance
                self.fuerzas[v] = self.fuerzas[v][0] + force[0], self.fuerzas[v][1] + force[1]

    def fruchterman_reingold_step(self):
        self.initialize_forces()
        self.compute_attraction_forces()
        self.compute_repulsion_forces()
        self.compute_gravity_forces()
        self.update_positions()
        self.temperature *= self.temperatureConstant
        self.print_message("Actualizando temperatura a " + str(self.temperature))


def leer_grafo(nombre_archivo):
    with open(nombre_archivo) as f:
        lines = f.readlines()
    vertices = []
    aristas = []
    if len(lines) > 0:
        n = lines[0].rstrip('\n').rstrip(' ')
        if n.isnumeric():
            n = int(n)
            if len(lines) < (n + 1):
                raise Exception('Cantidad invalida de vertices')
            for i in range(1, n + 1):
                line = lines[i].rstrip('\n')
                if line.count(' ') != 0:
                    raise Exception('Nombre de vertice invalido')
                vertices.append(line)
            for i in range(n + 1, len(lines)):
                line = lines[i].rstrip('\n')
                args = line.split(' ')
                if len(args) != 2 or vertices.count(args[0]) == 0 or vertices.count(args[1]) == 0:
                    raise Exception('Arista invalida')
                arista = (args[0], args[1])
                if aristas.count(arista) != 0:
                    raise Exception('Arista duplicada')
                if arista[0] == arista[1]:
                    raise Exception('Lazo no permitido')
                aristas.append(arista)
        else:
            raise Exception("Cantidad de vertices invalida")
    else:
        raise Exception('Archivo vacio')

    return vertices, aristas


def main():
    # Definimos los argumentos de linea de comando que aceptamos
    parser = argparse.ArgumentParser()

    # Verbosidad, opcional, False por defecto
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Muestra mas informacion al correr el programa'
    )
    # Cantidad de iteraciones, opcional, 50 por defecto
    parser.add_argument(
        '--iters',
        type=int,
        help='Cantidad de iteraciones a efectuar',
        default=50
    )
    # Temperatura inicial
    parser.add_argument(
        '--temp',
        type=float,
        help='Temperatura inicial',
        default=100.0
    )
    # Refresh
    parser.add_argument(
        '--ref',
        type=int,
        help='Tasa de refresco',
        default=1
    )
    # Constante de repulsion
    parser.add_argument(
        '--cr',
        type=float,
        help='Constante de repulsion',
        default=0.1
    )
    # Constante de atraccion
    parser.add_argument(
        '--ca',
        type=float,
        help='Constante de atraccion',
        default=5
    )
    # Constante de gravedad
    parser.add_argument(
        '--g',
        type=float,
        help='Constante de gravedad',
        default=3
    )
    # Constante de Enfriamiento
    parser.add_argument(
        '--cooling',
        type=float,
        help='Constante de enfriamiento',
        default=0.95
    )
    # Ancho de la pestaña
    parser.add_argument(
        '--width',
        type=int,
        help='Ancho de la pestaña',
        default=500
    )
    # Alto de la pestaña
    parser.add_argument(
        '--height',
        type=int,
        help='Alto de la pestaña',
        default=500
    )
    # Archivo del cual leer el grafo
    parser.add_argument(
        'file_name',
        help='Archivo del cual leer el grafo a dibujar'
    )
    parser.add_argument(
        '-s', '--save',
        help='Guarda el resultado en un archivo'
    )

    args = parser.parse_args()

    grafo = leer_grafo(args.file_name)

    # Creamos nuestro objeto LayoutGraph
    layout_gr = LayoutGraph(
        grafo,
        iters=args.iters,
        temperature=args.temp,
        refresh=args.ref,
        c1=args.cr,
        c2=args.ca,
        width=args.width,
        height=args.height,
        gravity=args.g,
        cooling=args.cooling,
        save=args.save,
        verbose=args.verbose
    )

    # Ejecutamos el layout
    layout_gr.layout()
    return


if __name__ == '__main__':
    main()
