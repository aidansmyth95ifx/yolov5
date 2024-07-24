import cv2

def plot_bboxes(img_file, bboxes, fmt='xyxy'):

    img = cv2.imread(img_file)
    #new_size = [int(x/2) for x in img.shape[:2]]
    #img = cv2.resize(img, (new_size[1],new_size[0]))
    h, w = img.shape[:2]
    print('{} {}'.format(w,h))

    for bbox in bboxes: # [cx, cy, w, h]
        # plot rectangle on img
        # Convert [cx, cy, w, h] to [x1, y1, x2, y2] format
        if fmt == 'cxcywh':
            x1 = int( w * (bbox[0] - (bbox[2]) / 2))  # Calculate top-left x coordinate
            y1 = int( h * (bbox[1] - (bbox[3]) / 2))  # Calculate top-left y coordinate
            x2 = int( w * (bbox[0] + (bbox[2]) / 2))  # Calculate bottom-right x coordinate
            y2 = int( h * (bbox[1] + (bbox[3]) / 2))  # Calculate bottom-right y coordinate
        elif fmt == 'xyxy':
            x1, y1, x2, y2 = bbox[:4]
        elif fmt == 'yxyx':
            y1, x1, y2, x2 = bbox[:4]
        else:
            raise Exception

        # Plot the bounding box on the image
        print('{} {} {} {}'.format(x1, y1, x2, y2))
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw the bounding
    cv2.imshow('Aligned image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    img_file = 'runs/detect/exp131/data_dump/image_3/image_pre_resize.jpg'
    bboxes = [
        [115, 77, 405, 230]
    ]

    plot_bboxes(img_file, bboxes, fmt='xyxy')
