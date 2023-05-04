import pygame

class Visualizer:

    def __init__(self, screen=None):  # constructor for tracker/calib mode
        self.screen = screen
        self.font = pygame.font.SysFont("Arial", 30)


    def fill_background(self, color_or_img):
        self.screen.fill(color_or_img)

    def draw_point(self, point, color, radius):
        pygame.draw.circle(self.screen, color, point, radius)

    def draw_text(self, text, color, pos):
        self.screen.blit(self.font.render(text, True, color), pos)

    def draw_ref_points(self, ref_points):
        for point in ref_points:
            self.draw_point(point, (0, 255, 0), 10)

    def draw_arrow(self, start, end, color, width=1):
        pygame.draw.line(self.screen, color, start, end, width)







