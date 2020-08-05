import os
import time
import neat
import pickle
import pygame
import numpy as np
from keras.utils import to_categorical

def init_screen(win):
    line_interval = 20
    line_color = (255,255,255)

    win.fill((0,0,0))
    for i in range(25):
        pygame.draw.line(win,line_color,(0,i*line_interval),(500,i*line_interval))
        pygame.draw.line(win,line_color,(i*line_interval,0),(i*line_interval,500))
# init_screen(win)

class food:
    def __init__(self, default_x = None, default_y = None):
        self.color = (0,255,0)
        if(default_x != None):
            self.x = default_x
        else:
            self.x = np.random.randint(0,25)
        if(default_y != None):
            self.y = default_y
        else:
            self.y = np.random.randint(0,25)

        self.width = 20
        self.height = 20
        self.eaten = False

    def spawn(self, win):
        pygame.draw.rect(win, self.color, (self.width * self.x, self.height*self.y,self.width,self.height))

class snake:
    def __init__(self):
        self.move_count = 0
        self.bite = False
        self.length = 1;
        self.body = [[np.random.randint(0,25),np.random.randint(0,25)]]
        self.width = 20
        self.height = 20
        self.velocity = 20
        self.color = (255,0,0)
        self.x_change = 1 # Heading right
        self.y_change = 0 # Not changing

    def check_collide(self):
        return (self.body[0] in self.body[1::]) or self.body[0][0] < 0 or\
            self.body[0][1] < 0 or self.body[0][0] == 25 or self.body[0][1] == 25

    def move(self, win, key):
        self.move_count += 1

        if np.array_equal(key ,[1, 0, 0, 0]) and self.y_change is not 1: # moving top
            move_array = [0,-1]
        elif np.array_equal(key, [0,1,0,0]) and self.y_change is not -1:  # moving bottom
            move_array = [0,1]
        elif np.array_equal(key, [0, 0, 1, 0]) and self.x_change is not -1:  # moving right
            move_array = [1, 0]
        elif np.array_equal(key, [0, 0, 0, 1]) and self.x_change is not 1:  # moving bottom
            move_array = [-1, 0]
        else : move_array = [self.x_change, self.y_change]
        self.x_change, self.y_change = move_array

        self.body.insert(0, [(self.body[0][0] + self.x_change), (self.body[0][1] + self.y_change)])
        self.body.pop(len(self.body) - 1)

        for node in self.body:
            if (node == self.body[0]):
                if (self.x_change == 0 and self.y_change < 0):
                    pygame.draw.rect(win, (0, 0, 0), (node[0] * 20 + 4, node[1] * 20, 4, 4))
                    pygame.draw.rect(win, (0, 0, 0), (node[0] * 20 + 12, node[1] * 20, 4, 4))
                if (self.x_change == 0 and self.y_change > 0):
                    pygame.draw.rect(win, (0, 0, 0), (node[0] * 20 + 4, node[1] * 20 + 16, 4, 4))
                    pygame.draw.rect(win, (0, 0, 0), (node[0] * 20 + 12, node[1] * 20 + 16, 4, 4))
                if (self.x_change > 0 and self.y_change == 0):
                    pygame.draw.rect(win, (0, 0, 0), (node[0] * 20 + 16, node[1] * 20 + 4, 4, 4))
                    pygame.draw.rect(win, (0, 0, 0), (node[0] * 20 + 16, node[1] * 20 + 12, 4, 4))
                if (self.x_change < 0 and self.y_change == 0):
                    pygame.draw.rect(win, (0, 0, 0), (node[0] * 20, node[1] * 20 + 4, 4, 4))
                    pygame.draw.rect(win, (0, 0, 0), (node[0] * 20, node[1] * 20 + 12, 4, 4))

                pygame.draw.rect(win, (255, 255, 0), (node[0] * 20, node[1] * 20, self.width, self.height))
            else:
                pygame.draw.rect(win, self.color, (node[0] * 20, node[1] * 20, self.width, self.height))

    def spawn(self, win):
        pygame.draw.rect(win, (255,255,0), (self.body[0][0]*20, self.body[0][1]*20, self.width, self.height))


def get_state(snake, food):

        snake_x = snake.body[0][0]
        snake_y = snake.body[0][1]

        food_x = food.x
        food_y = food.y

        state = [
            # danger top:
            ((snake.x_change == 0 and snake.y_change == -1 and (snake_y - 1 < 0 or [snake.body[0][0] + 0, snake.body[0][
                1] - 1] in snake.body)) or  # heading top and saw your tail or wall
             (snake.x_change == 1 and snake.y_change == 0 and (snake_y - 1 < 0 or [snake.body[0][0] + 0, snake.body[0][
                 1] - 1] in snake.body)) or  # heading right and on top there is ur tail and wall,
             (snake.x_change == -1 and snake.y_change == 0 and (
                         snake_y - 1 < 0 or [snake.body[0][0] + 0, snake.body[0][1] - 1] in snake.body))),
            # danger bottom
            ((snake.x_change == 0 and snake.y_change == 1 and (
                        snake_y + 1 == 25 or [snake.body[0][0] + 0, snake.body[0][1] + 1] in snake.body)) or
             (snake.x_change == 1 and snake.y_change == 0 and (
                         snake_y + 1 == 25 or [snake.body[0][0] + 0, snake.body[0][1] + 1] in snake.body)) or
             (snake.x_change == -1 and snake.y_change == 0 and (
                         snake_y + 1 == 25 or [snake.body[0][0] + 0, snake.body[0][1] + 1] in snake.body))),
            # danger right
            ((snake.x_change == 1 and snake.y_change == 0 and (
                        snake_x + 1 == 25 or [snake.body[0][0] + 1, snake.body[0][1] + 0] in snake.body)) or
             (snake.x_change == 0 and snake.y_change == -1 and (
                         snake_x + 1 == 25 or [snake.body[0][0] + 1, snake.body[0][1] + 0] in snake.body)) or
             (snake.x_change == 0 and snake.y_change == 1 and (
                         snake_x + 1 == 25 or [snake.body[0][0] + 1, snake.body[0][1] + 0] in snake.body))),
            # danger left
            ((snake.x_change == -1 and snake.y_change == 0 and (
                        snake_x - 1 < 0 or [snake.body[0][0] - 1, snake.body[0][1] + 0] in snake.body)) or
             (snake.x_change == 0 and snake.y_change == -1 and (
                         snake_x - 1 < 0 or [snake.body[0][0] - 1, snake.body[0][1] + 0] in snake.body)) or
             (snake.x_change == 0 and snake.y_change == 1 and (
                         snake_x - 1 < 0 or [snake.body[0][0] - 1, snake.body[0][1] + 0] in snake.body))),

            snake.x_change == 1,  # moving right
            snake.x_change == -1,  # moving left
            snake.y_change == 1,  # moving down
            snake.y_change == -1,  # moving up

            food_x > snake_x,  # food at the right
            food_x < snake_x,  # food at the left
            food_y > snake_y,  # food at the bottom
            food_y < snake_y  # food at the top
        ]
        for i in range(len(state)):
            if (state[i]):
                state[i] = 1
            else:
                state[i] = 0
        return tuple(state)

FIRST_ITERATION = True
def eval_genomes(genomes, config):
    global FIRST_ITERATION
    snakes = [] # A list to hold all the Snake objects
    ge = [] # A list to hold all the genomes
    nets = [] # A list to hold all the networks corresponding their genomes

    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        snake_ = snake()
        snake_.spawn(win)

        nets.append(net)
        snakes.append(snake_)

        if(os.path.exists("genome.pickle") and FIRST_ITERATION):
           genome = pickle.load(open("genome.pickle", "rb"))

        FIRST_ITERATION = False

        ge.append(genome)

    # move the snakes inside the array first:
    run =  True
    food_ = food()

    while(run):
        # Problem : some individual are moving back and forth
        time.sleep(0.01)
        pygame.time.delay(0)
        init_screen(win)
        food_.spawn(win)
        for x, snake_ in enumerate(snakes):
            last_tail = [snake_.body[0][0], snake_.body[0][1]]
            state = get_state(snake_, food_)

            keys = nets[x].activate(state)
            final_move = to_categorical(np.argmax(keys), num_classes=4)

            snake_.move(win, final_move)

            if (food_.x == snake_.body[0][0] and food_.y == snake_.body[0][1]):
                food_ = food()  # if food is eaten, get new food
                snake_.body.append(last_tail)

                print("Food eaten")
                ge[snakes.index(snake_)].fitness += 2
                snake_.move_count = 0

            if((snake_.move_count > 100 and food_.eaten == False) or snake_.check_collide()):
                ge[snakes.index(snake_)].fitness -= 2
                if(snake_.move_count > 100 and food_.eaten == False):
                    ge[snakes.index(snake_)].fitness = -3
              
                ge.pop(snakes.index(snake_))
                nets.pop(snakes.index(snake_))
                snakes.pop(snakes.index(snake_))

        for event in pygame.event.get():
            if(event.type == pygame.QUIT):
                pygame.display.quit()
                run = False

        pygame.display.update()
        for genome in ge:
            genome.fitness += 0.1 # fitness increase by 0.1 for any frame it survive

        if(len(snakes) == 0): run  = False

def run(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())

    winner = p.run(eval_genomes, n = 1000)
    pickle.dump(winner, open("genome.pickle", "wb"))

    print("Best genome : {}\n".format(winner))


if __name__ == '__main__':
    pygame.init()
    win = pygame.display.set_mode((500, 500))
    pygame.display.set_caption("Snake ML")

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')

    run(config_path)
