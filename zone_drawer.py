# zone_drawer.py
import cv2
import json
import numpy as np
import os

class ZoneDrawer:
    def __init__(self):
        self.zones = {
            'lanes': [],        # List of lane polygons
            'intersection': [],   # Intersection polygon
            'traffic_lights': []  # Traffic light points [{x, y, color}, ...]
        }
        self.current_points = []
        self.current_mode = None  # 'lane', 'intersection', or 'traffic_light'
        self.current_light_color = None  # 'red' or 'green'
    
    def draw_zones(self, video_path, zones_file='zones.json'):
        """
        Interactive tool to draw lanes and intersection on first frame
        """
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("Error reading video")
            return
        
        self.frame = frame.copy()
        
        # Load existing zones if file exists
        if os.path.exists(zones_file):
            try:
                with open(zones_file, 'r') as f:
                    self.zones = json.load(f)
                # Ensure all required keys exist
                self.zones.setdefault('lanes', [])
                self.zones.setdefault('intersection', [])
                self.zones.setdefault('traffic_lights', [])
                print(f"Loaded existing zones: {len(self.zones['lanes'])} lanes, intersection: {len(self.zones['intersection'])} points, {len(self.zones['traffic_lights'])} lights")
            except Exception as e:
                print(f"Error loading zones: {e}")
                self.zones = {'lanes': [], 'intersection': [], 'traffic_lights': []}
        
        print("\n=== Zone Drawing Tool ===")
        print("1. Draw LANES: Press '1' to start drawing lane polygons")
        print("2. Draw INTERSECTION: Press '2' to start drawing intersection polygon")
        print("3. Draw GREEN LIGHT: Press '3' to click green light position")
        print("4. Draw RED LIGHT: Press '4' to click red light position")
        print("\nControls:")
        print("  - Click to add points (or finalize light)")
        print("  - Press 'c' to complete polygon")
        print("  - Press 's' to save")
        print("  - Press 'r' to reset current")
        print("  - Press 'd' to delete last lane/intersection/light")
        print("  - Press 'ESC' to quit\n")
        
        cv2.namedWindow('Zone Drawing')
        cv2.setMouseCallback('Zone Drawing', self.mouse_callback)
        
        while True:
            display = self.frame.copy()
            self.draw_current_zones(display)
            
            # Draw mode indicator
            if self.current_mode:
                cv2.putText(display, f"Mode: {self.current_mode.upper()}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Zone Drawing', display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord('1'):
                self.current_mode = 'lane'
                self.current_points = []
                print("Mode: LANE - Click to add points, 'c' to complete")
            elif key == ord('2'):
                self.current_mode = 'intersection'
                self.current_points = []
                print("Mode: INTERSECTION - Click to add points, 'c' to complete")
            elif key == ord('3'):
                self.current_mode = 'traffic_light'
                self.current_light_color = 'green'
                self.current_points = []
                print("Mode: GREEN LIGHT - Click on green light position")
            elif key == ord('4'):
                self.current_mode = 'traffic_light'
                self.current_light_color = 'red'
                self.current_points = []
                print("Mode: RED LIGHT - Click on red light position")
            elif key == ord('c'):
                self.complete_polygon()
            elif key == ord('s'):
                self.save_zones(zones_file)
                print("Zones saved!")
            elif key == ord('r'):
                self.current_points = []
                print("Current points reset")
            elif key == ord('d'):
                self.delete_last()
        
        cv2.destroyAllWindows()
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_mode is None:
                print("Select mode: Press '1' for lane, '2' for intersection")
                return
            
            if self.current_mode == 'traffic_light':
                # Add traffic light point
                self.zones['traffic_lights'].append({'x': x, 'y': y, 'color': self.current_light_color})
                print(f"Traffic light ({self.current_light_color}) added at ({x}, {y})")
            else:
                self.current_points.append((x, y))
                print(f"Point added: ({x}, {y}). Points: {len(self.current_points)}")
    
    def complete_polygon(self):
        if self.current_mode is None:
            print("Select mode first")
            return
        
        if self.current_mode == 'traffic_light':
            print(f"Traffic lights set: {len(self.zones['traffic_lights'])} points")
            if len(self.zones['traffic_lights']) >= 2:
                print("Both red and green lights defined")
            return
        
        if len(self.current_points) < 3:
            print(f"Need at least 3 points. Current: {len(self.current_points)}")
            return
        
        if self.current_mode == 'lane':
            self.zones['lanes'].append(self.current_points.copy())
            print(f"Lane {len(self.zones['lanes'])} completed with {len(self.current_points)} points")
        elif self.current_mode == 'intersection':
            self.zones['intersection'] = self.current_points.copy()
            print(f"Intersection completed with {len(self.current_points)} points")
        
        self.current_points = []
    
    def delete_last(self):
        if self.current_mode == 'lane' and self.zones['lanes']:
            removed = self.zones['lanes'].pop()
            print(f"Deleted last lane. Remaining: {len(self.zones['lanes'])}")
        elif self.current_mode == 'intersection' and self.zones['intersection']:
            self.zones['intersection'] = []
            print("Deleted intersection")
        elif self.current_mode == 'traffic_light' and self.zones['traffic_lights']:
            removed = self.zones['traffic_lights'].pop()
            print(f"Deleted traffic light at ({removed['x']}, {removed['y']}) - {removed['color']}")
    
    def draw_current_zones(self, display):
        # Draw lanes
        for i, lane in enumerate(self.zones['lanes']):
            if len(lane) >= 3:
                pts = np.array(lane, dtype=np.int32)
                cv2.polylines(display, [pts], True, (0, 255, 255), 2)  # Yellow
                # Label lane
                cv2.putText(display, f"Lane {i+1}", tuple(lane[0]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw intersection
        if len(self.zones['intersection']) >= 3:
            pts = np.array(self.zones['intersection'], dtype=np.int32)
            cv2.polylines(display, [pts], True, (255, 0, 0), 2)  # Blue
            cv2.putText(display, "Intersection", tuple(self.zones['intersection'][0]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw traffic light points
        for tl in self.zones['traffic_lights']:
            color = (0, 0, 255) if tl['color'] == 'red' else (0, 255, 0)  # Red or Green
            cv2.circle(display, (tl['x'], tl['y']), 8, color, -1)
            cv2.circle(display, (tl['x'], tl['y']), 10, color, 2)
            label = 'R' if tl['color'] == 'red' else 'G'
            cv2.putText(display, label, (tl['x'] - 5, tl['y'] + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw current points being drawn for lanes/intersection
        for p in self.current_points:
            cv2.circle(display, p, 4, (0, 255, 0), -1)  # Green
        
        if len(self.current_points) >= 2 and self.current_mode != 'traffic_light':
            pts = np.array(self.current_points, dtype=np.int32)
            cv2.polylines(display, [pts], False, (0, 255, 0), 1)
    
    def save_zones(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.zones, f, indent=2)