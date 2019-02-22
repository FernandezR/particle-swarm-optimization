import copy
import numpy as np


class Function(object):
    f = None
    a = None
    b = None

    @staticmethod
    def set_func(func):
        Function.f = func
        Function.a, Function.b = {
            Function.sphere: (-5.12, 5.12),
            Function.rastrigin: (-5.12, 5.12),
            Function.rosenbrock: (-2.048, 2.048),
            Function.ackley: (-32, 32)
        }.get(func, None)
        pass

    @staticmethod
    def sphere(arg_vec):
        return np.sum([x ** 2 for x in arg_vec])

    @staticmethod
    def rosenbrock(arg_vec):
        return sum([(100 * (xj - xi ** 2) ** 2 + (xi - 1) ** 2) for xi, xj in zip(arg_vec[:-1], arg_vec[1:])])

    @staticmethod
    def rastrigin(arg_vec):
        return 10 * len(arg_vec) + np.sum([x ** 2 - 10 * np.cos(2 * np.pi * x) for x in arg_vec])

    @staticmethod
    def ackley(arg_vec):
        s1 = -0.2 * np.sqrt(np.sum([x ** 2 for x in arg_vec]) / len(arg_vec))
        s2 = np.sum([np.cos(2 * np.pi * x) for x in arg_vec]) / len(arg_vec)
        return 20 + np.e - 20 * np.exp(s1) - np.exp(s2)


class Particle:
    def __init__(self, dim, l_bound, u_bound):
        self.position = PSO.float_rand(l_bound, u_bound, dim)
        self.velocity = PSO.float_rand(l_bound, u_bound, dim)
        self.fitness = Function.f(self.position)
        self.best_part_pos = copy.copy(self.position)
        self.best_part_fitness = self.fitness


class PSO(object):
    def __init__(self,
                 num_iterations,
                 num_particles,
                 dim,
                 l_bound,
                 u_bound,
                 inert,
                 cognitive,
                 social):
        self.num_iterations = num_iterations
        self.num_particles = num_particles
        self.dim = dim
        self.l_bound = l_bound
        self.u_bound = u_bound
        self.inertia = inert
        self.cognitive = cognitive
        self.social = social
        self.swarm = [Particle(dim, l_bound, u_bound) for _ in range(num_particles)]
        self.best_swarm_position = copy.copy(sorted(self.swarm, key=lambda x: x.fitness)[0].position)
        self.best_swarm_fitness = sorted(self.swarm, key=lambda x: x.fitness)[0].fitness

    def run(self):
        for i in range(1, self.num_iterations + 1):
            for j in range(self.num_particles):
                # compute new velocity of curr particle
                for k in range(self.dim):
                    new_velocity = ((self.inertia * self.swarm[j].velocity[k]) +
                                    (self.cognitive * np.random.random() *
                                     (self.swarm[j].best_part_pos[k] - self.swarm[j].position[k])) +
                                    (self.social * np.random.random() * (self.best_swarm_position[k] -
                                                                         self.swarm[j].position[k])))
                    self.swarm[j].velocity[k] = max(min(new_velocity, self.u_bound), self.l_bound)

                # compute new position using new velocity
                self.swarm[j].position += self.swarm[j].velocity
                # compute error of new position
                self.swarm[j].fitness = Function.f(self.swarm[j].position)

                # is new position a new best for the particle?
                if self.swarm[j].fitness < self.swarm[j].best_part_fitness:
                    self.swarm[j].best_part_fitness = self.swarm[j].fitness
                    self.swarm[j].best_part_pos = copy.copy(self.swarm[j].position)

                # is new position a new best overall?
                if self.swarm[j].fitness < self.best_swarm_fitness:
                    self.best_swarm_fitness = self.swarm[j].fitness
                    self.best_swarm_position = copy.copy(self.swarm[j].position)

            # for-each particle
            if i % 10 == 0:
                print('{0} {1}'.format(i, self.best_swarm_fitness))

    @staticmethod
    def float_rand(a, b, size=None):
        return a + ((b - a) * np.random.random(size))


dimension = 5
particles = 500
iterations = 1000
inertia = 0.729
cognitive_particle = 1.49445
social_swarm = 1.49445
Function.set_func(Function.rosenbrock)


def main():
    pso = PSO(iterations, particles, dimension, Function.a, Function.b, inertia, cognitive_particle, social_swarm)
    pso.run()
    print('\nBest solution found: {0}'.format(pso.best_swarm_position))
    print('Error of best solution = {0}'.format(pso.best_swarm_fitness))


if __name__ == '__main__':
    main()
