import pydicom
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.draw import polygon, polygon_perimeter
import plotly.graph_objects as go
from skimage import measure
import json
from matplotlib.widgets import Button

class DICOMLabeler:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.data_zyx = self.load_dicom_series()
        self.slice_labels = np.zeros_like(self.data_zyx, dtype=np.int32)
        self.current_slice = 0
        self.polygons = []
        self.current_polygon = []
        self.lines = []
        self.bone_colors = {
            1: ('red', 'Femur'),
            2: ('blue', 'Tibia'),
            3: ('green', 'Fibula'),
            4: ('yellow', 'Patella'),
            5: ('purple', 'Other Bone 1'),
            6: ('orange', 'Other Bone 2')
        }
        self.waiting_for_label = False
        
    def load_dicom_series(self):
        slices = []
        for filename in sorted(os.listdir(self.folder_path)):
            if filename.endswith('.dcm'):
                filepath = os.path.join(self.folder_path, filename)
                ds = pydicom.dcmread(filepath)
                slices.append(ds.pixel_array)
        return np.stack(slices, axis=0)
    
    def start_labeling(self):
        self.setup_ui()
        plt.show()
        
    def setup_ui(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.subplots_adjust(bottom=0.25)
        
        # Add buttons
        ax_prev = plt.axes([0.2, 0.05, 0.15, 0.075])
        ax_next = plt.axes([0.4, 0.05, 0.15, 0.075])
        ax_finish = plt.axes([0.6, 0.05, 0.15, 0.075])
        ax_save = plt.axes([0.4, 0.15, 0.2, 0.075])
        
        self.btn_prev = Button(ax_prev, 'Previous Slice')
        self.btn_next = Button(ax_next, 'Next Slice')
        self.btn_finish = Button(ax_finish, 'Finish Polygon')
        self.btn_save = Button(ax_save, 'Save Labels')
        
        self.btn_prev.on_clicked(lambda event: self.prev_slice())
        self.btn_next.on_clicked(lambda event: self.next_slice())
        self.btn_finish.on_clicked(lambda event: self.initiate_finish_polygon())
        self.btn_save.on_clicked(lambda event: self.save_labels())
        
        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        self.update_display()
        
    def update_display(self):
        self.ax.clear()
        self.ax.imshow(self.data_zyx[self.current_slice], cmap='gray')
        
        for poly_points, label in self.polygons:
            poly = np.array(poly_points)
            rr, cc = polygon_perimeter(poly[:, 1], poly[:, 0], self.data_zyx.shape[1:])
            color = self.bone_colors.get(label, ('white', 'Unknown'))[0]
            self.ax.plot(cc, rr, color=color, linewidth=2)
            
            centroid = poly.mean(axis=0)
            self.ax.text(centroid[0], centroid[1], str(label), 
                        color=color, fontsize=12, weight='bold')
        
        if len(self.current_polygon) > 1:
            x, y = zip(*self.current_polygon)
            self.ax.plot(x, y, 'r-', linewidth=2)
        
        self.ax.set_title(f"Slice {self.current_slice+1}/{self.data_zyx.shape[0]}\n"
                         "Left click: Add point | Right click/Enter/Button: Finish polygon\n"
                         "Delete/Backspace: Remove last point | 'c': Clear current")
        self.fig.canvas.draw()
        
    def on_click(self, event):
        if self.waiting_for_label:
            return
            
        print(f"Button pressed: {event.button}")  # Debug output
        if event.inaxes != self.ax:
            return
            
        if event.button == 1:  # Left click
            self.current_polygon.append((event.xdata, event.ydata))
            self.update_display()
        else:  # Treat any other click as right-click
            self.initiate_finish_polygon()
            
    def on_key_press(self, event):
        if self.waiting_for_label:
            return
            
        if event.key in ['delete', 'backspace']:
            if self.current_polygon:
                self.current_polygon.pop()
                self.update_display()
        elif event.key == 'c':
            self.current_polygon = []
            self.update_display()
        elif event.key == 'enter':
            self.initiate_finish_polygon()
            
    def initiate_finish_polygon(self):
        """Start the polygon finishing process"""
        if self.waiting_for_label:
            return
            
        if len(self.current_polygon) < 3:
            print("Need at least 3 points to form a polygon.")
            return
            
        self.waiting_for_label = True
        self.fig.canvas.draw()  # Update display before blocking
        
        # Create a small popup window for label input
        import tkinter as tk
        from tkinter import simpledialog
        
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        print("\nAvailable bone labels:")
        for label, (color, name) in self.bone_colors.items():
            print(f"{label}: {name} ({color})")
            
        label = simpledialog.askinteger("Bone Label", "Enter bone label number:", parent=root)
        if label is not None and label in self.bone_colors:
            self.polygons.append((self.current_polygon.copy(), label))
            self.current_polygon = []
        
        self.waiting_for_label = False
        self.update_display()
        
    def prev_slice(self):
        if self.current_slice > 0:
            self.save_current_slice()
            self.current_slice -= 1
            self.load_slice_data()
            self.update_display()
            
    def next_slice(self):
        if self.current_slice < self.data_zyx.shape[0] - 1:
            self.save_current_slice()
            self.current_slice += 1
            self.load_slice_data()
            self.update_display()
            
    def save_current_slice(self):
        label_array = np.zeros(self.data_zyx.shape[1:], dtype=np.int32)
        for poly_points, label in self.polygons:
            poly = np.array(poly_points)
            rr, cc = polygon(poly[:, 1], poly[:, 0], self.data_zyx.shape[1:])
            label_array[rr, cc] = label
        self.slice_labels[self.current_slice] = label_array
        
    def load_slice_data(self):
        self.polygons = []
        label_array = self.slice_labels[self.current_slice]
        
        for label in np.unique(label_array):
            if label == 0:
                continue
                
            contours = measure.find_contours(label_array == label, 0.5)
            for contour in contours:
                if len(contour) > 100:
                    from skimage.measure import approximate_polygon
                    contour = approximate_polygon(contour, tolerance=1)
                
                poly_points = [(pt[1], pt[0]) for pt in contour]
                self.polygons.append((poly_points, label))
                
    def save_labels(self):
        self.save_current_slice()
        np.save('bone_labels.npy', self.slice_labels)
        
        metadata = {
            'bone_colors': {k: v[0] for k, v in self.bone_colors.items()},
            'bone_names': {k: v[1] for k, v in self.bone_colors.items()}
        }
        with open('label_metadata.json', 'w') as f:
            json.dump(metadata, f)
            
        print("Labels saved to bone_labels.npy and label_metadata.json")
        
    def visualize_3d(self):
        self.save_current_slice()
        fig = go.Figure()
        
        for label in np.unique(self.slice_labels):
            if label == 0:
                continue
                
            bone_mask = (self.slice_labels == label).astype(np.uint8)
            verts, faces, _, _ = measure.marching_cubes(bone_mask, level=0.5)
            
            color = self.bone_colors.get(label, ('gray', 'Unknown'))[0]
            name = self.bone_colors.get(label, ('gray', 'Unknown'))[1]
            
            fig.add_trace(go.Mesh3d(
                x=verts[:, 2], y=verts[:, 1], z=verts[:, 0],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color=color,
                opacity=0.7,
                name=f'{label}: {name}'
            ))
            
        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            title="3D Visualization of Labeled Bones",
            width=1000,
            height=800
        )
        
        fig.show()

if __name__ == "__main__":
    folder_path = 'knee2_mini'  # Change to your DICOM folder
    labeler = DICOMLabeler(folder_path)
    
    print("Starting labeling tool...")
    print("Multiple ways to finish a polygon:")
    print("- Right-click (if working on your system)")
    print("- Press Enter key")
    print("- Click 'Finish Polygon' button")
    
    labeler.start_labeling()
    print("\nLabeling complete. Generating 3D visualization...")
    labeler.visualize_3d()