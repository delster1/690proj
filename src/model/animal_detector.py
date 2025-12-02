import cv2
import numpy as np
import tensorflow as tf

class YOLOv4TFLiteDetector:
    def __init__(self, model_path, class_names=None, input_size=416, 
                 conf_threshold=0.5, nms_threshold=0.4):
        """
        Initialize YOLOv4 TFLite detector.
        
        Args:
            model_path: Path to YOLOv4 TFLite model (.tflite)
            class_names: List of class names or path to .names file
            input_size: Input size (YOLOv4 typically uses 416, 512, or 608)
            conf_threshold: Confidence threshold for detections
            nms_threshold: Non-maximum suppression threshold
        """
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        # Load class names
        if isinstance(class_names, str):
            with open(class_names, 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]
        elif class_names:
            self.class_names = class_names
        else:
            self.class_names = [f"Class_{i}" for i in range(80)]  # COCO default
        
        # Generate random colors for each class
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.class_names), 3), dtype=np.uint8)
        
        # Load TFLite model
        self.load_tflite_model(model_path)
    
    def load_tflite_model(self, model_path):
        """Load TFLite YOLOv4 model."""
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"[loaded] TFLite model from {model_path}")
        print(f"[info] Input shape: {self.input_details[0]['shape']}")
        print(f"[info] Input dtype: {self.input_details[0]['dtype']}")
        print(f"[info] Number of outputs: {len(self.output_details)}")
        for i, detail in enumerate(self.output_details):
            print(f"[info] Output {i} shape: {detail['shape']}, dtype: {detail['dtype']}")
        
        # Check if model is quantized
        input_scale, input_zero_point = self.input_details[0]['quantization']
        self.is_quantized = input_scale > 0
        if self.is_quantized:
            print(f"[info] Model is quantized (INT8)")
        else:
            print(f"[info] Model is floating point (FP32)")
    
    def preprocess_frame(self, frame):
        """Preprocess frame for YOLOv4."""
        # Resize to input size
        resized = cv2.resize(frame, (self.input_size, self.input_size))
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Add batch dimension
        batched = np.expand_dims(normalized, axis=0)
        
        return batched
    
    def detect(self, frame):
        """Run object detection on frame."""
        height, width = frame.shape[:2]
        
        # Preprocess
        input_data = self.preprocess_frame(frame)
        
        # Handle quantization if needed
        if self.is_quantized:
            input_scale, input_zero_point = self.input_details[0]['quantization']
            input_data = input_data / input_scale + input_zero_point
            input_data = input_data.astype(np.int8)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Get outputs
        outputs = []
        for detail in self.output_details:
            output = self.interpreter.get_tensor(detail['index'])
            
            # Dequantize if needed
            output_scale, output_zero_point = detail['quantization']
            if output_scale > 0:
                output = (output.astype(np.float32) - output_zero_point) * output_scale
            
            outputs.append(output)
        
        # Parse detections
        detections = self.parse_outputs(outputs, width, height)
        
        return detections
    
    def parse_outputs(self, outputs, width, height):
        """
        Parse TFLite model outputs.
        
        Common YOLOv4 TFLite output formats:
        Format 1: [boxes, classes, scores, num_detections]
        Format 2: [boxes, scores, classes, num_detections]
        Format 3: Single output with shape [1, num_detections, 6] (x, y, w, h, conf, class)
        """
        detections = []
        
        # Detect output format based on shapes
        if len(outputs) == 4:
            # Format 1 or 2: Multiple outputs
            boxes = outputs[0][0]
            
            # Determine if format 1 or 2 based on dtype
            if outputs[1].dtype in [np.int32, np.int64]:
                # Format 1: [boxes, classes, scores, num]
                classes = outputs[1][0]
                scores = outputs[2][0]
            else:
                # Format 2: [boxes, scores, classes, num]
                scores = outputs[1][0]
                classes = outputs[2][0]
            
            num_detections = int(outputs[3][0][0]) if outputs[3].ndim > 1 else int(outputs[3][0])
            
            for i in range(num_detections):
                if scores[i] > self.conf_threshold:
                    # Boxes format: [ymin, xmin, ymax, xmax] normalized
                    ymin, xmin, ymax, xmax = boxes[i]
                    x = int(xmin * width)
                    y = int(ymin * height)
                    w = int((xmax - xmin) * width)
                    h = int((ymax - ymin) * height)
                    
                    class_id = int(classes[i])
                    if class_id < len(self.class_names):
                        detections.append({
                            'box': [x, y, w, h],
                            'confidence': float(scores[i]),
                            'class_id': class_id,
                            'class_name': self.class_names[class_id]
                        })
        
        elif len(outputs) == 1:
            # Format 3: Single output
            output = outputs[0][0]  # Remove batch dimension
            
            for detection in output:
                if len(detection) == 6:
                    # Format: [x_center, y_center, width, height, confidence, class_id]
                    x_center, y_center, w_norm, h_norm, conf, class_id = detection
                    
                    if conf > self.conf_threshold:
                        x = int((x_center - w_norm / 2) * width)
                        y = int((y_center - h_norm / 2) * height)
                        w = int(w_norm * width)
                        h = int(h_norm * height)
                        
                        class_id = int(class_id)
                        if class_id < len(self.class_names):
                            detections.append({
                                'box': [x, y, w, h],
                                'confidence': float(conf),
                                'class_id': class_id,
                                'class_name': self.class_names[class_id]
                            })
                elif len(detection) >= 5:
                    # Format: [x, y, w, h, conf, class_scores...]
                    x_center = detection[0]
                    y_center = detection[1]
                    w_norm = detection[2]
                    h_norm = detection[3]
                    conf = detection[4]
                    
                    if conf > self.conf_threshold:
                        class_scores = detection[5:]
                        class_id = np.argmax(class_scores)
                        
                        x = int((x_center - w_norm / 2) * width)
                        y = int((y_center - h_norm / 2) * height)
                        w = int(w_norm * width)
                        h = int(h_norm * height)
                        
                        if class_id < len(self.class_names):
                            detections.append({
                                'box': [x, y, w, h],
                                'confidence': float(conf * class_scores[class_id]),
                                'class_id': class_id,
                                'class_name': self.class_names[class_id]
                            })
        
        # Apply NMS to remove overlapping boxes
        detections = self.apply_nms(detections)
        
        return detections
    
    def apply_nms(self, detections):
        """Apply Non-Maximum Suppression to detections."""
        if len(detections) == 0:
            return []
        
        boxes = [det['box'] for det in detections]
        confidences = [det['confidence'] for det in detections]
        
        # Convert to format expected by cv2.dnn.NMSBoxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 
                                    self.conf_threshold, self.nms_threshold)
        
        filtered_detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                filtered_detections.append(detections[i])
        
        return filtered_detections
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels on frame."""
        for det in detections:
            x, y, w, h = det['box']
            class_id = det['class_id']
            confidence = det['confidence']
            label = det['class_name']
            
            # Get color for this class
            color = tuple(map(int, self.colors[class_id]))
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Prepare label text
            text = f"{label}: {confidence:.2f}"
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            # Draw background rectangle for text
            cv2.rectangle(frame, (x, y - text_height - 10), 
                         (x + text_width, y), color, -1)
            
            # Draw text
            cv2.putText(frame, text, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def run(self, camera_id=0, display_fps=True, save_video=None):
        """
        Run real-time object detection on webcam feed.
        
        Args:
            camera_id: Camera device ID (0 for default webcam)
            display_fps: Whether to display FPS counter
            save_video: Path to save output video (optional)
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_id}")
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Setup video writer if saving
        writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_video, fourcc, fps, 
                                    (frame_width, frame_height))
            print(f"[info] Saving video to {save_video}")
        
        print("[info] Starting webcam... Press 'q' to quit")
        
        # FPS calculation
        fps_counter = 0
        fps_start_time = cv2.getTickCount()
        current_fps = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[error] Failed to grab frame")
                break
            
            # Run detection
            detections = self.detect(frame)
            
            # Draw results
            frame = self.draw_detections(frame, detections)
            
            # Draw detection count
            cv2.putText(frame, f"Detections: {len(detections)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Calculate and display FPS
            if display_fps:
                fps_counter += 1
                if fps_counter % 10 == 0:
                    fps_end_time = cv2.getTickCount()
                    time_diff = (fps_end_time - fps_start_time) / cv2.getTickFrequency()
                    current_fps = 10 / time_diff
                    fps_start_time = fps_end_time
                
                cv2.putText(frame, f"FPS: {current_fps:.1f}", 
                           (frame_width - 150, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display instructions
            cv2.putText(frame, "Press 'q' to quit", (10, frame_height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Save frame if recording
            if writer:
                writer.write(frame)
            
            # Show frame
            cv2.imshow('YOLOv4 TFLite Object Detection', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("[info] Webcam stopped")


def main():
    # Configure your YOLOv4 TFLite model here
    MODEL_PATH = "fake_yolov4.tflite"
    CLASS_NAMES = "fake_classes.names"  # or ['cat', 'dog', 'bird'] as list
    INPUT_SIZE = 416  # Must match your model's input size
    CONF_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.4
    
    # Create detector
    detector = YOLOv4TFLiteDetector(
        model_path=MODEL_PATH,
        class_names=CLASS_NAMES,
        input_size=INPUT_SIZE,
        conf_threshold=CONF_THRESHOLD,
        nms_threshold=NMS_THRESHOLD
    )
    
    # Run on webcam
    detector.run(camera_id=0, display_fps=True)
    
    # Optional: Save output video
    # detector.run(camera_id=0, display_fps=True, save_video="output.mp4")


if __name__ == "__main__":
    main()