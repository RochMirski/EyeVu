import numpy as np
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

"""f_m = 10
f_e = 18
f_c = 3.04
p=2
x_1 = 2
L_c = 5
a = 2.76
L = 70
D = 24
L_1 = 20

x_cent = -f_e + D/2"""

def plot_vertical_line(x, h, **kwargs):
        """
        Plot a vertical line of height h at position x.

        Parameters:
        x (float): The x-coordinate of the line
        h (float): The height of the line (extends from 0 to h)
        **kwargs: Additional arguments passed to plt.plot (e.g., color, linestyle)
        """
        ax = kwargs.pop('ax', plt.gca())
        ax.plot([x, x], [-h/2, h/2], **kwargs)

def plot_circle(x, y, D, **kwargs):
        """
        Plot a circle centered at (x, y) with radius r.

        Parameters:
        x, y (float): Center coordinates of the circle
        r (float): Radius of the circle
        **kwargs: Additional keyword arguments for styling (e.g., color, linewidth, fill)
        """
        ax = kwargs.pop('ax', plt.gca())
        circle = plt.Circle((x, y), D/2, fill=False, **kwargs)
        ax.add_patch(circle)

def lens_forward(focal_length, h, grad):
  s = h/grad

  d = 1/(1/focal_length - 1/s)

  grad = -h/d

  return d, grad

def lens_backward(focal_length, h, grad):

  d = -h/grad

  s = 1/(1/focal_length - 1/d)

  grad = h/s

  return -s, grad

def calc_eye_pos(x_int, grad, x_cent, D):
    a = 1 + grad**2
    b = -2 * (x_cent + grad**2 * x_int)
    c = x_cent**2 + (grad*x_int)**2-(D**2)/4
    determinant = b**2 - 4*a*c
    if determinant < 0:
        return
    x_circ = (-b - np.sqrt(determinant))/(2*a)
    y_circ = -grad*(x_int-x_circ)

    return [x_circ, y_circ]


def half_angle_theta_from_FOV(FOV, D):
   return FOV/D

def angular_eye_pos(theta, D, x_cent):
   x = x_cent - D/2 * np.cos(theta)
   y = -D/2 * np.sin(theta)
   return x, y

"""FOV = 15
lenses = [f_e, f_m, f_c]
separations = [L_1, L, L_c]
ang = half_angle_theta_from_FOV(FOV, D)
first_coord = angular_eye_pos(half_angle_theta_from_FOV(FOV, D), D, x_cent)
plt.scatter(first_coord[0], first_coord[1]) """

def single_ray_plot(h_e1, lenses, separations, first_coord):

    #plt.scatter(x_cent, 0)

    #print(s_e2 + d_e2 - separations[0])
    next_x = 0
    next_y = h_e1
    grad = (h_e1-first_coord[1])/-first_coord[0]

    #temp = lens_backward(f_e, h_e1, grad)


    points = [first_coord, [next_x, next_y]]



    for i in range(len(lenses)):
        d, grad = lens_forward(lenses[i], next_y, grad)

        next_x = next_x + separations[i]

        next_y =  next_y + grad * separations[i]

        points.append([next_x, next_y])
    print(points)

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    

    return xs, ys

"""h_e1 = -0.1

xs, ys = single_ray_plot(h_e1, lenses, separations, first_coord)

plt.plot(xs, ys)

for i in range(0, len(lenses)+2):
    plot_vertical_line(xs[i], [0, 12, 60, 10, 2.76][i])

h_e1 = -0.75
xs, ys = single_ray_plot(h_e1, lenses, separations, first_coord)

plt.plot(xs, ys, 'r')

plot_circle(x_cent, 0, D)
plt.scatter(x_1, p/2)
plt.scatter(x_1, -p/2)
plt.grid(True)
plt.show()"""

#plt.ion()  # Turn on interactive mode
#plt.show()

# --- Tkinter UI Setup ---
class OpticsCalcApp:
    def __init__(self, master):
        self.master = master
        master.title("Optics Calc Interactive Plot")

        # Default values
        self.params = {
            'f_m': 40,
            'f_e': 18,
            'f_c': 3.04,
            'p': 1.5,
            'x_1': 2,
            'L_c': 4,
            'a': 2.76,
            'L': 39,
            'D': 24,
            'L_1': 40,
            'FOV': 4.6,
            'h_e1_ray1': 0.4,
            'h_e1_ray2': -0.95
        }
        self.sliders = {}
        self.create_widgets()
        self.draw_plot()

    def create_widgets(self):
        # Sliders frame
        sliders_frame = tk.Frame(self.master)
        sliders_frame.pack(side=tk.LEFT, fill=tk.Y)
        slider_specs = [
            ('f_m', 1, 50, 0.1),
            ('f_e', 1, 50, 0.1),
            ('f_c', 0.1, 20, 0.01),
            ('p', 0.1, 10, 0.01),
            ('x_1', -20, 20, 0.1),
            ('L_c', 1, 50, 0.1),
            ('a', 0.1, 10, 0.01),
            ('L', 1, 200, 0.1),
            ('D', 1, 50, 0.1),
            ('L_1', 1, 50, 0.1),
            ('FOV', 1, 90, 0.1),
            ('h_e1_ray1', -5, 5, 0.01),
            ('h_e1_ray2', -5, 5, 0.01)
        ]
        for i, (name, minv, maxv, res) in enumerate(slider_specs):
            label = tk.Label(sliders_frame, text=name)
            label.grid(row=i, column=0, sticky='w')
            slider = tk.Scale(sliders_frame, from_=minv, to=maxv, resolution=res, orient=tk.HORIZONTAL, length=180,
                             command=lambda val, n=name: self.on_slider_change(n, val))
            slider.set(self.params[name])
            slider.grid(row=i, column=1)
            self.sliders[name] = slider

        # Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(7, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)

        # Add zoom button
        zoom_btn = tk.Button(self.master, text="Toggle Zoom", command=self.toggle_zoom)
        zoom_btn.pack(side=tk.BOTTOM, fill=tk.X)

    def on_slider_change(self, name, val):
        self.params[name] = float(val)
        self.draw_plot()

    def draw_plot(self):
        self.ax.clear()
        # Unpack parameters
        f_m = self.params['f_m']
        f_e = self.params['f_e']
        f_c = self.params['f_c']
        p = self.params['p']
        x_1 = self.params['x_1']
        L_c = self.params['L_c']
        a = self.params['a']
        L = self.params['L']
        D = self.params['D']
        L_1 = self.params['L_1']
        FOV = self.params['FOV']
        h_e1_ray1 = self.params['h_e1_ray1']
        h_e1_ray2 = self.params['h_e1_ray2']
        h_e1_ray3 = (h_e1_ray1 + h_e1_ray2) / 2

        x_cent = -f_e + D/2
        lenses = [f_e, f_m, f_c]
        separations = [L_1, L, L_c]
        ang = half_angle_theta_from_FOV(FOV, D)
        first_coord = angular_eye_pos(half_angle_theta_from_FOV(FOV, D), D, x_cent)
        self.ax.scatter(first_coord[0], first_coord[1], color='k', label='Eye')

        # Ray 1
        xs1, ys1 = single_ray_plot(h_e1_ray1, lenses, separations, first_coord)
        self.ax.plot(xs1, ys1, label='Ray 1')
        # Ray 2
        xs2, ys2 = single_ray_plot(h_e1_ray2, lenses, separations, first_coord)
        self.ax.plot(xs2, ys2, 'r', label='Ray 2')
        # Ray 3 (average)
        xs3, ys3 = single_ray_plot(h_e1_ray3, lenses, separations, first_coord)
        self.ax.plot(xs3, ys3, 'g', label='Ray 3 (avg)')

        # Vertical lines
        heights = [0, 12, 60, 10, a]
        for i in range(0, len(lenses)+2):
            plot_vertical_line([xs1, xs2, xs3][0][i], heights[i], ax=self.ax)

        # Circle
        plot_circle(x_cent, 0, D, ax=self.ax)
        self.ax.scatter(x_1, p/2, color='g')
        self.ax.scatter(x_1, -p/2, color='g')

        # Grid settings
        self.ax.grid(True, which='major', linestyle='-', linewidth=0.8)
        self.ax.minorticks_on()
        self.ax.grid(True, which='minor', linestyle=':', linewidth=0.5)
        self.ax.legend()
        self.ax.set_title("Optics Ray Plot")

        # Zoom/scale logic
        if hasattr(self, 'zoomed') and self.zoomed:
            # Do not enforce equal aspect when zoomed
            pass
        else:
            self.ax.set_aspect('equal', adjustable='datalim')
            # Set limits to fit all rays and circle
            all_xs = xs1 + xs2 + xs3 + [x_cent, x_1]
            all_ys = ys1 + ys2 + ys3 + [p/2, -p/2, 0]
            margin = 0.1 * max(D, L, L_1)
            self.ax.set_xlim(min(all_xs)-margin, max(all_xs)+margin)
            self.ax.set_ylim(min(all_ys)-margin, max(all_ys)+margin)
        self.canvas.draw()

    def toggle_zoom(self):
        self.zoomed = not getattr(self, 'zoomed', False)
        self.draw_plot()

# --- Launch Tkinter app ---
if __name__ == "__main__":
    root = tk.Tk()
    app = OpticsCalcApp(root)
    root.mainloop()