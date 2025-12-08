from qdrant_client import models


def show_results(points: list[models.ScoredPoint]) -> None:
    print("# results")
    for point in points:
        print(f"- id: {point.id}, score: {point.score}, text: {point.payload['text']}")
    print()
