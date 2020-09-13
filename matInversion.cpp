#include <bits/stdc++.h>
#include <mpi.h>

using namespace std;


MPI_Comm my_comm;

const double ONE = 1.000;
void print_matrix(double *matrix, int rows, int cols, int rank);
void init_matrix(double *matrix, double *eye, int n);
void transpose_matrix(double *matrix, int n);
void print_matrix(double *matrix, int rows, int cols);
void preprocess_matrix(double *matrix, double *eye, int n);
void swap_rows(double *row1, double *row2, int start1, int start2, int cols);
bool local_row(int rank, int global_pos, int n, int procs);
void eliminate_column(double *sub_matrix, double *sub_eye, double *row, double *eye_row, int startr, int endr, int pivot_column, int cols);


struct double_int {
	double val;
	int pos;
};
typedef struct double_int double_int; 


double_int find_local_maxima(double *matrix, int startr, int endr, int cols, int rank, int procs, int pivot);

void mult_row(double *matrix, int start, int cols, double num)
{
	for(int i = 0 ; i < cols ; ++i)
	{
		matrix[start*cols + i] = matrix[start*cols + i]*num;
	}
}

void mult_mat(double *res, double *mat1, double *mat2, int n)
{
	for(int i  = 0 ; i < n ; ++i)
	{
		for(int j = 0 ; j < n ; ++j)
		{
			res[i*n+j] = 0;
			for(int k = 0 ; k < n ; ++k)
			{
				res[i*n+j] += mat1[i*n+k]*mat2[k*n+j];
			}
		}
	}
}

void copy_mat(double *src, double *dest, int rows, int cols)
{
	for(int i = 0 ; i < rows ; ++i)
	{
		for(int j = 0 ; j < cols ; ++j)
		{
			dest[i*cols+j] = src[i*cols+j];
		}
	}
}

void swap_routine(double *sub_matrix, double *row, double *swap_row, int rank, int procs, int N, int swap_pos, int cur_pos)
{
	int num_rows = N/procs;
	int start_row = num_rows*rank;

	if(min(swap_pos / num_rows, procs-1) == rank)
	{
		memcpy(swap_row, &sub_matrix[(swap_pos - start_row)*N], N*sizeof(double));
	}
	if(min(cur_pos/num_rows , procs - 1) == rank)
	{
		memcpy(row, &sub_matrix[(cur_pos - start_row)*N], N*sizeof(double));
	}

	MPI_Bcast(swap_row, N, MPI_DOUBLE, min(swap_pos / num_rows, procs-1), my_comm);
	MPI_Bcast(row, N, MPI_DOUBLE, min(cur_pos/num_rows, procs-1), my_comm);

	if(min(swap_pos / num_rows, procs-1) == rank)
	{
		swap_rows(row, sub_matrix, 0, swap_pos - start_row, N);
	}
	if(min(cur_pos/num_rows , procs - 1) == rank)
	{
		swap_rows(swap_row, sub_matrix, 0, cur_pos - start_row, N);
	}
} 



void forward_elimination(double *matrix, double *eye, int rank, int procs, int N, bool do_swap);

int main(int argc, char *argv[])
{
	int rank, procs;
	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &procs);

	int N;
	double *matrix;
	double *original;
	double *eye;

	if(rank == 0)
	{
		cin >> N;
		matrix = new double [N*N];
		original = new double [N*N];
		eye = new double [N*N];
		init_matrix(matrix, eye, N);
		copy_mat(matrix, original, N, N);
	//	print_matrix(matrix, N, N);
	}
	
	MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
	int color = rank / N;

	MPI_Comm_split(MPI_COMM_WORLD, color, rank, &my_comm);
	MPI_Comm_rank(my_comm, &rank);
	MPI_Comm_size(my_comm, &procs);
	if(color == 0)
		forward_elimination(matrix, eye, rank, procs, N, true);
	
    	
	//backsubstitution
	double scale;
	if(rank == 0 && color == 0)
	{
//		print_matrix(eye, N, N);
		for(int i = N-1 ; i > 0 ; --i)
		{
			for(int j = 0 ; j < i ; ++j)
			{
				scale = matrix[j*N+i];
				for(int k = 0 ; k < N ; ++k)
				{
					matrix[j*N+k] -= scale*matrix[i*N+k];
					eye[j*N+k] -= scale*eye[i*N+k];
				}
			}
		}
	}

	MPI_Finalize();

	if(rank == 0 && color == 0)
	{
		
		//cout << endl << "inv(matrix) == " << endl;
		print_matrix(eye, N, N);
	//	cout << endl << " mat == " << endl;
	//	print_matrix(matrix, N, N);
		/*double *res = new double [N*N];
		
		cout << "inv(matrix) * matrix ==" << endl;
		mult_mat(res, eye, original, N);
		print_matrix(res, N, N);
		

		cout << "matrix * inv(matrix) ==" << endl;
		mult_mat(res, original, eye, N);
		print_matrix(res, N, N);
		
		delete[] res;*/
	
	}
	if(rank == 0 && color == 0)
	{
		delete[] matrix;
		delete[] eye;
		delete[] original;
	}
	return 0;
}
void print_matrix(double *matrix, int rows, int cols)
{
	for(int i = 0 ; i < rows ; ++i)
	{
		for(int j = 0 ; j < cols ; ++j)
		{
			cout << matrix[i*cols+j] << " ";
		}
		cout << endl;
	}
}

void print_matrix(double *matrix, int rows, int cols, int rank)
{
	for(int i = 0 ; i < rows ; ++i)
	{
		cout << "rank == " << rank << " ";
		for(int j = 0 ; j < cols ; ++j)
		{
			cout << matrix[i*cols+j] << " ";
		}
		cout << endl;
	}
}

void init_matrix(double *matrix, double *eye, int n)
{
	for(int i = 0 ; i < n ; ++i)
	{
		for(int j = 0 ; j < n ; ++j)
		{
			cin >> matrix[i*n+j];
			if(i == j)
			{
				eye[i*n+j] = 1.0;
			}
			else
			{
				eye[i*n+j] = 0.0;
			}
		}
	}   
}


void swap_rows(double *row1, double  *row2, int start1, int start2, int cols)
{
	for(int i = 0 ; i < cols ; ++i)
	{
		double temp = row1[start1*cols+i];
		row1[start1*cols+i] = row2[start2*cols+i];
		row2[start2*cols+i] = temp;
	}

}


double_int find_local_maxima(double *matrix, int startr, int endr, int cols, int rank, int procs, int pivot)
{
	int num_rows = cols/procs;
	double_int local_max;
	local_max.pos = startr;
	local_max.val = 0;
	for(int i = startr ; i < endr ; ++i)
	{
		double temp = abs(matrix[i*cols + pivot]);
		if(local_max.val <= temp)
		{
			local_max.val = temp;
			local_max.pos = i;
		}
	}
	local_max.pos += rank*num_rows;
	return local_max;
}

void eliminate_column(double *sub_matrix, double *sub_eye, double *row, double *eye_row, int startr, int endr, int pivot_column, int cols)
{
	for(int j = startr ; j < endr ; ++j)
	{
		double scale = sub_matrix[j*cols+pivot_column];
		for(int k = 0 ; k < cols ; ++k)
		{
			sub_matrix[j*cols+k] -= scale*row[k];
			sub_eye[j*cols+k] -= scale*eye_row[k];
		}
	}
}
void forward_elimination(double *matrix, double *eye, int rank, int procs, int N, bool do_swap)
{
	MPI_Barrier(my_comm);
		
	int num_rows = N/procs;

	int *send_counts = new int[procs];
	
	int *displs = new int[procs];

	for(int i = 0 ; i < procs; ++i)
	{
		send_counts[i] = N*num_rows;
		displs[i] = i*num_rows*N;
	}
	send_counts[procs-1] += ((N%procs)*N);

	double *sub_matrix = new double [N * (num_rows+N%procs)];
	double *sub_eye = new double [N* (num_rows+N%procs)];

	MPI_Scatterv(matrix, send_counts, displs, MPI_DOUBLE, sub_matrix, send_counts[rank], MPI_DOUBLE, 0, my_comm);
   	MPI_Scatterv(eye, send_counts, displs, MPI_DOUBLE, sub_eye, send_counts[rank], MPI_DOUBLE, 0, my_comm);

	double *row = new double[N];
	double *swap_row = new double[N];
	double *eye_row = new double[N];

	double pivot;
	int start_row;
	double_int local_max;
	double_int global_max;
	MPI_Barrier(my_comm); 

	start_row = rank*num_rows;

	//number of rows distributed to this process
	int my_size = num_rows + (rank == procs -1 ? N%procs : 0);
	for(int i = 0 ; i < N ; ++i)
	{
		//Note:
		// 'i' is the pivot column
		// sender of 'i' pivot will be process with rank i/num_rows
		// all others will receive

		//synchronise all processes
		MPI_Barrier(my_comm);
	
		//get local maximma for ith column:
		//	finds maximma in my range, if pivot lies ahead of my range then gives 0 as val ( participates as dummy)

		local_max = find_local_maxima(sub_matrix, max(0, i-start_row), my_size, N, rank, procs, i);				
		MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE_INT, MPI_MAXLOC, my_comm);

		//the swap routine
		//broadcasts both rows required for swapping
		//processes check for themselves if they have to perform swapping

		// Note:
		// process with max val in pivot column is global_max.pos / num_rows
		// swaps will be with ith row
		swap_routine(sub_matrix, row, swap_row, rank, procs, N, global_max.pos, i);
		swap_routine(sub_eye, row, swap_row, rank, procs, N, global_max.pos, i);	

		//normalisation step of gaussan elimination	
		if(i>=start_row && i < start_row+my_size)
		{		
			pivot = sub_matrix[(i-start_row)*N+i];
			
			mult_row(sub_matrix, i-start_row, N, ONE/pivot);
			mult_row(sub_eye, i-start_row, N, ONE/pivot);			
					
			memcpy(row, &sub_matrix[(i-start_row)*N], N*sizeof(double));
			memcpy(eye_row, &sub_eye[(i-start_row)*N], N*sizeof(double));

		}	
		
		//broadcasting normalised row to all
		MPI_Bcast(row, N, MPI_DOUBLE, min(i/num_rows,procs-1), my_comm);
		MPI_Bcast(eye_row, N, MPI_DOUBLE, min(i/num_rows,procs-1), my_comm);

		//elimination step of gaussian elimination
		eliminate_column(sub_matrix, sub_eye, row, eye_row, max(0, i+1 - start_row), my_size, i, N);
		
	}

	MPI_Gatherv(sub_matrix, send_counts[rank], MPI_DOUBLE, matrix, send_counts, displs, MPI_DOUBLE, 0, my_comm);
	MPI_Gatherv(sub_eye, send_counts[rank], MPI_DOUBLE, eye, send_counts, displs, MPI_DOUBLE, 0, my_comm);

	delete[] sub_matrix;
	delete[] sub_eye;
	delete[] row;
	delete[] eye_row;
	delete[] swap_row;
	delete[] send_counts;
	delete[] displs;

}



