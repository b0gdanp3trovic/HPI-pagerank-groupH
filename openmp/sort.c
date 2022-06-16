
// using OMI
#include <omp.h>

// Function to swap two numbers a and b
void swap(int* a, int* b)
{
	int t = *a;
	*a = *b;
	*b = t;
}

// Function to perform the partitioning
// of array arr[]
int partition(int arr[], int arr2[], int start, int end)
{
	// Declaration
	int pivot = arr[end];
	int i = (start - 1);

	// Rearranging the array
	for (int j = start; j <= end - 1; j++) {
		if (arr[j] < pivot) {
			i++;
			swap(&arr[i], &arr[j]);
            swap(&arr2[i], &arr2[j]);
		}
	}
	swap(&arr[i + 1], &arr[end]);
    swap(&arr2[i + 1], &arr2[end]);

	// Returning the respective index
	return (i + 1);
}

// Function to perform QuickSort Algorithm
// using openmp
void quicksort(int arr[], int arr2[], int start, int end)
{
	// Declaration
	int index;

	if (start < end) {

		// Getting the index of pivot
		// by partitioning
		index = partition(arr, arr2, start, end);

        // Parallel sections
        //#pragma omp parallel sections
		{
            //#pragma omp section
			{
				// Evaluating the left half
				quicksort(arr, arr2, start, index - 1);
			}
            //#pragma omp section
			{
				// Evaluating the right half
				quicksort(arr, arr2, index + 1, end);
			}
		}
	}
}

