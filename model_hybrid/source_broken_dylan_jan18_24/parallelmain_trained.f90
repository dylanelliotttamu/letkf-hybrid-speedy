program main
  use mpi_f08
  use, intrinsic :: ieee_arithmetic

  use mpires, only : mpi_res, startmpi, distribute_prediction_marker, killmpi, predictionmpicontroller
  use mod_reservoir, only : initialize_model_parameters, allocate_res_new, train_reservoir, start_prediction, initialize_prediction, predict
  use speedy_res_interface, only : startspeedy
  use resdomain, only : processor_decomposition, initializedomain, set_reservoir_by_region
  use mod_utilities, only : main_type, init_random_seed, dp, gaussian_noise, standardize_data_given_pars4d, standardize_data_given_pars3d, standardize_data, init_random_marker
  !use mod_unit_tests, only : test_linalg, test_res_domain #TODO not working yet

  implicit none 

  integer :: standardizing_vars, i, j, t, prediction_num

  logical :: runspeedy = .False. 

  type(main_type) :: res

  !Fortran has command line augs TODO 

  !Starts the MPI stuff and initializes mpi_res
  call startmpi()
  
  !mpi_res%numprocs = 1152 
  !res%model_parameters%numprocs = 1152 

  !Makes the object called res and declares all of the main parameters 
  call initialize_model_parameters(res%model_parameters,mpi_res%proc_num,mpi_res%numprocs)

  !Do domain decomposition based off processors and do vertical localization of
  !reservoir
  call processor_decomposition(res%model_parameters)

  !Need this for each worker gets a new random seed
  !call init_random_seed(mpi_res%proc_num)
  call init_random_marker(33)

  !TODO Bug some where below this comment  
  do i=1,res%model_parameters%num_of_regions_on_proc
     do j=1,res%model_parameters%num_vert_levels
        call initialize_prediction(res%reservoir(i,j),res%model_parameters,res%grid(i,j))  
     enddo 
  enddo 

  do prediction_num=1, res%model_parameters%num_predictions
     do t=1, res%model_parameters%predictionlength/res%model_parameters%timestep
        if(t == 1) then 
          do i=1, res%model_parameters%num_of_regions_on_proc
             do j=1,res%model_parameters%num_vert_levels
                print *, 'starting start_prediction region',res%model_parameters%region_indices(i),'prediction_num prediction_num',prediction_num
                call start_prediction(res%reservoir(i,j),res%model_parameters,res%grid(i,j),prediction_num)
             enddo
          enddo 
        endif
        do i=1, res%model_parameters%num_of_regions_on_proc
           do j=1, res%model_parameters%num_vert_levels
              print *, 'calling predict'
              call predict(res%reservoir(i,j),res%model_parameters,res%grid(i,j),res%reservoir(i,j)%saved_state,res%reservoir(i,j)%local_model)
           enddo
        enddo
        print *, 'calling predictionmpicontroller','prediction_num prediction_num',prediction_num,'time',t
        call predictionmpicontroller(res,t)
      enddo
  enddo 

  call MPI_Barrier(mpi_res%mpi_world, mpi_res%ierr)

  call mpi_finalize(mpi_res%ierr)

  if(res%model_parameters%irank == 0) then
     print *, 'program finished correctly'
  endif   

end program

