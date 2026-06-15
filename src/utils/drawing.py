import cv2


def draw_face(
    frame,
    face
):

    x1, y1, x2, y2 = face.bbox.astype(int)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
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


def draw_registration_status(
    frame,
    registrar
):

    if not registrar.active:
        return

    text = (
        f"Registering "
        f"{registrar.name}: "
        f"{registrar.samples_collected}"
    )

    cv2.putText(
        frame,
        text,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

def draw_recognition(
    frame,
    face,
    name,
    score
):

    x1, y1, x2, y2 = face.bbox.astype(int)

    label = (
        f"{name} "
        f"({score:.2f})"
    )

    cv2.putText(
        frame,
        label,
        (x1, y1 - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )