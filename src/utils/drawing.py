import cv2


def draw_face(frame, face):

    x1, y1, x2, y2 = face.bbox.astype(int)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"{face.det_score:.2f}",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    for point in face.kps:

        x, y = point.astype(int)

        cv2.circle(
            frame,
            (x, y),
            3,
            (0, 0, 255),
            -1
        )