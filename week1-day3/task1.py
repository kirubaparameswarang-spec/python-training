import numpy as np

marks = np.array([
    [70, 80, 90],
    [60, 75, 85],
    [88, 92, 79],
    [55, 65, 70]
])

print("Shape:", marks.shape)

student_avg = np.mean(marks, axis=1)
print("Student averages:", np.round(student_avg, 2))

subject_avg = np.mean(marks, axis=0)
print("Subject averages:", subject_avg)

weights = np.array([0.3, 0.3, 0.4])
weighted_scores = np.dot(marks, weights)
print("Weighted scores:", weighted_scores)