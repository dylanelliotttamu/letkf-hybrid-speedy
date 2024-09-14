program main
  !use mpi_f08
  use mpi
  use, intrinsic :: ieee_arithmetic

  use mpires, only : mpi_res, startmpi, distribute_prediction_marker, killmpi, &
                     sendrecievegrid, distribute_speedy_forecast, stop_mpi_safe, distribute_ic_and_speedy_forecast, clean_up_speedy
  use mod_reservoir, only : initialize_model_parameters, allocate_res_new, train_reservoir, start_prediction, &
                            initialize_prediction, predict, trained_reservoir_prediction, predict_ml
  use mod_slab_ocean_reservoir, only : initialize_slab_ocean_model, train_slab_ocean_model, &
                                       get_training_data_from_atmo, initialize_prediction_slab, start_prediction_slab, &
                                       predict_slab, predict_slab_ml, trained_ocean_reservoir_prediction
  use speedy_res_interface, only : startspeedy
  use resdomain, only : processor_decomposition, initializedomain, set_reservoir_by_region
  use mod_utilities, only : main_type, init_random_seed, dp, gaussian_noise, standardize_data_given_pars4d, standardize_data_given_pars3d, &
                            standardize_data, init_random_marker, xgrid, ygrid, zgrid
  use mod_calendar
  use speedy_main, only : agcm_main
  use mod_physvar, only : ug1, vg1, tg1, qg1, phig1, pslg1, &
                          ug1_ic, vg1_ic, tg1_ic, qg1_ic, phig1_ic, pslg1_ic
  !use mod_unit_tests, only : test_linalg, test_res_domain #TODO not working yet

  implicit none 

  integer :: standardizing_vars, i, j, k , t, prediction_num
  integer :: year_i, month_i, day_i, hour_i 
  integer :: num_ensembles, ensemble_mem
  integer :: left_over, mem_counter

  logical :: runspeedy = .False.
  logical :: trained_model = .True.
  logical :: slab_model
 
  real(kind=dp), allocatable :: grid4d(:,:,:,:), grid2d(:,:), forecast_grid4d(:,:,:,:), forecast_grid2d(:,:)

  CHARACTER(100)     :: date, num_ensembles_char
  CHARACTER(len=255) :: root_path, current_path 
  character(len=3)   :: ensemble_path 
  character(len=:), allocatable :: ensemble_path_full

  type(main_type) :: res

  !Starts the MPI stuff and initializes mpi_res
  call startmpi()

  ! Read requested number of trials from command line arguments
  if (command_argument_count() /= 2) then
        write(0,"(A)") "Required argument: date"
        call stop_mpi_safe()
        stop 1
  end if

  call get_command_argument(1, date)

  read(date(1:4),'(I4)') year_i
  read(date(5:6),'(I2)') month_i
  read(date(7:8),'(I2)') day_i
  read(date(9:10),'(I2)') hour_i 

  call get_command_argument(2, num_ensembles_char)

  read(num_ensembles_char(1:3),'(I3)') num_ensembles

  call getcwd(root_path) 

  print *, 'root_path',trim(root_path)

  !This is a place holder doesnt mean anything  
  res%model_parameters%ensemble_member = 1

  !mpi_res%numprocs = 1152 

  !Makes the object called res and declares all of the main parameters 
  call initialize_model_parameters(res%model_parameters,mpi_res%proc_num,mpi_res%numprocs)

  !Do domain decomposition based off processors and do vertical localization of
  !reservoir
  call processor_decomposition(res%model_parameters)

  !Need this for each worker gets a new random seed
  call init_random_marker(33)


  !Allocate atmo3d reservoirs and any special ones 
  allocate(res%reservoir(res%model_parameters%num_of_regions_on_proc,res%model_parameters%num_vert_levels))
  allocate(res%grid(res%model_parameters%num_of_regions_on_proc,res%model_parameters%num_vert_levels))

  if(res%model_parameters%slab_ocean_model_bool) then
    res%model_parameters%special_reservoirs = .True.
    res%model_parameters%num_special_reservoirs = 1
  endif 

  !NOTE one day may make precip its own special reservoir 
  !if(res%model_parameters%precip_bool) then
  !  res%model_parameters%num_special_reservoirs = res%model_parameters%num_special_reservoirs + 1
  !endif

  if(res%model_parameters%special_reservoirs) then
    allocate(res%reservoir_special(res%model_parameters%num_of_regions_on_proc,res%model_parameters%num_special_reservoirs))
    allocate(res%grid_special(res%model_parameters%num_of_regions_on_proc,res%model_parameters%num_special_reservoirs))
  endif  


  !---This is for debugging----!
  !You can run the code with a small number of processors and look at a few
  !regions of the globe 
  !if(res%model_parameters%irank == 4) res%model_parameters%region_indices(1) = 954
  !if(res%model_parameters%irank == 2) res%model_parameters%region_indices(1) = 552
  !if(res%model_parameters%irank == 3)  res%model_parameters%region_indices(1) = 36

  !If we already trained and are just reading in files then we go here 
  if(trained_model) then
    !Loop 1: Loop over all sub domains (regions) on each processor
    print *, 'res%model_parameters%num_of_regions_on_proc',res%model_parameters%num_of_regions_on_proc
    do i=1,res%model_parameters%num_of_regions_on_proc
       print *, 'i', i
       !Loop 2: Loop over each vertical level for a particular sub domain
        do j=1,res%model_parameters%num_vert_levels
           print *, 'j', j
           print *, 'doing initializedomain'
           call initializedomain(res%model_parameters%number_of_regions,res%model_parameters%region_indices(i), &
                                 res%model_parameters%overlap,res%model_parameters%num_vert_levels,j,res%model_parameters%vert_loc_overlap, &
                                 res%grid(i,j))


           res%reservoir(i,j)%assigned_region = res%model_parameters%region_indices(i)
           res%grid(i,j)%level_index = j

           print *, 'doing trained_reservoir_prediction'

           call initialize_calendar(calendar,1981,1,1,0)

           call trained_reservoir_prediction(res%reservoir(i,j),res%model_parameters,res%grid(i,j))
            
        enddo
  
        !Lets read in special reservoir 
        if(res%model_parameters%slab_ocean_model_bool) then 
          call initializedomain(res%model_parameters%number_of_regions,res%model_parameters%region_indices(i), &
                             res%model_parameters%overlap,res%model_parameters%num_vert_levels,j-1,res%model_parameters%vert_loc_overlap, &
                             res%grid_special(i,1))


          res%grid_special(i,1)%level_index = j-1

          res%reservoir_special(i,1)%assigned_region = res%model_parameters%region_indices(i)
 
          call trained_ocean_reservoir_prediction(res%reservoir_special(i,1),res%model_parameters,res%grid_special(i,1),res%reservoir(i,j-1),res%grid(i,j-1))
        endif 
     end do
     print *, 'done reading trained model'
  endif  
    
  !Initialize Prediction 
  !Loop through all of the regions and vertical levels 
 
  do i=1,res%model_parameters%num_of_regions_on_proc
     do j=1,res%model_parameters%num_vert_levels
        print *,'initialize prediction region,level',res%reservoir(i,j)%assigned_region,res%grid(i,j)%level_index
        call initialize_prediction(res%reservoir(i,j),res%model_parameters,res%grid(i,j))  
     enddo 

     if(res%model_parameters%slab_ocean_model_bool) then
        if(res%reservoir_special(i,1)%sst_bool_prediction) then
          print *,'ocean model initialize prediction region,i',res%reservoir_special(i,1)%assigned_region,i
          print *, 'shape(res%reservoir_special)',shape(res%reservoir_special)
          call initialize_prediction_slab(res%reservoir_special(i,1),res%model_parameters,res%grid_special(i,1),res%reservoir(i,j-1),res%grid(i,j-1))
        endif
     endif 
  enddo 

  do ensemble_mem=1, num_ensembles

  write(ensemble_path,'(i0.3)') ensemble_mem

  ensemble_path_full = trim(root_path)//'/'//ensemble_path
  print *, 'ensemble_path_full',ensemble_path_full

  res%model_parameters%ensemble_member = ensemble_mem

  CALL chdir(ensemble_path_full)

  call getcwd(current_path)

  print *, 'current path', trim(current_path)

  !NOTE first loop through the ensemble members SPEEDY forecasts 
  !If(ensemble_mem = mpires%irank) then run speedy
  !Make getcwd depend on processor rankk
  if(mpi_res%is_root) then
    call agcm_main()
  endif

  if(mpi_res%is_root) then
    !Need to make this for both the ICs and the SPEEDY forecast
    if(.not. allocated(grid4d))           allocate(grid4d(res%model_parameters%full_predictvars,xgrid,ygrid,zgrid))
    if(.not. allocated(grid2d))           allocate(grid2d(xgrid,ygrid))
    if(.not. allocated(forecast_grid4d))  allocate(forecast_grid4d(res%model_parameters%full_predictvars,xgrid,ygrid,zgrid))
    if(.not. allocated(forecast_grid4d))  allocate(forecast_grid2d(xgrid,ygrid))

    grid4d(1,:,:,:) = reshape(tg1_ic,[xgrid,ygrid,zgrid])
    grid4d(2,:,:,:) = reshape(ug1_ic,[xgrid,ygrid,zgrid])
    grid4d(3,:,:,:) = reshape(vg1_ic,[xgrid,ygrid,zgrid])
    grid4d(4,:,:,:) = reshape(qg1_ic,[xgrid,ygrid,zgrid])

    grid2d = reshape(pslg1_ic,[xgrid,ygrid])

    forecast_grid4d(1,:,:,:) = reshape(tg1,[xgrid,ygrid,zgrid])
    forecast_grid4d(2,:,:,:) = reshape(ug1,[xgrid,ygrid,zgrid])
    forecast_grid4d(3,:,:,:) = reshape(vg1,[xgrid,ygrid,zgrid])
    forecast_grid4d(4,:,:,:) = reshape(qg1,[xgrid,ygrid,zgrid])

    forecast_grid2d = reshape(pslg1,[xgrid,ygrid])
   
    call distribute_ic_and_speedy_forecast(res,grid4d,grid2d,forecast_grid4d,forecast_grid2d)
  else
    call distribute_ic_and_speedy_forecast(res)
  endif 

  !END ensemble loop
  !MPI_barrier
  !NOTE 
  
  !Begin ensemble loop again 
  !If(ensemble_member == irank) then 
  ! calldistribute_ic_and_speedy_forecast(res,grid4d,grid2d,forecast_grid4d,forecast_grid2d)
  !else
  !  call distribute_ic_and_speedy_forecast(res)
  !endif


  !Main prediction loop. 
  !Loop 1 through the user specified number of predictions
  !Loop 2 through time for a specific prediction
  !Loop 3/4 over the number of regions on the processor and all of the vertical
  !levels for a region
  do prediction_num=1, res%model_parameters%num_predictions
     do t=1, res%model_parameters%predictionlength/res%model_parameters%timestep
        if(t == 1) then 
          do i=1, res%model_parameters%num_of_regions_on_proc
             do j=1,res%model_parameters%num_vert_levels
                if(res%reservoir(i,j)%assigned_region == 954) print *, 'starting start_prediction region',res%model_parameters%region_indices(i),'prediction_num prediction_num',prediction_num
                call start_prediction(res%reservoir(i,j),res%model_parameters,res%grid(i,j),prediction_num)
             enddo
             if(res%model_parameters%slab_ocean_model_bool) then
               if(res%reservoir_special(i,1)%sst_bool_prediction) then 
                 call start_prediction_slab(res%reservoir_special(i,1),res%model_parameters,res%grid_special(i,1),res%reservoir(i,j-1),res%grid(i,j-1),prediction_num) 
               endif 
             endif 
          enddo 
        endif
        do i=1, res%model_parameters%num_of_regions_on_proc
           do j=1, res%model_parameters%num_vert_levels
              if(res%reservoir(i,j)%assigned_region == 954) print *, 'calling predict'
              if(res%model_parameters%ml_only) then
                call predict_ml(res%reservoir(i,j),res%model_parameters,res%grid(i,j),res%reservoir(i,j)%saved_state)
                res%model_parameters%run_speedy = .True.
              else
                call predict(res%reservoir(i,j),res%model_parameters,res%grid(i,j),res%reservoir(i,j)%saved_state,res%reservoir(i,j)%local_model)
              endif 
           enddo
           !print *, 'mod((t-1)*res%model_parameters%timestep,res%model_parameters%timestep_slab)',mod((t-1)*res%model_parameters%timestep,res%model_parameters%timestep_slab)
           if(res%model_parameters%slab_ocean_model_bool) then
             if(mod((t)*res%model_parameters%timestep,res%model_parameters%timestep_slab) == 0 .and. res%reservoir_special(i,1)%sst_bool_prediction) then
                if(res%reservoir_special(i,1)%assigned_region == 954) print *, 'calling predict slab'
                !TODO rolling_average_over_a_period(grid,period)
                !if( t > 28) then
                !  res%reservoir_special(i,1)%local_model = res%reservoir_special(i,1)%outvec 
                !endif 
                if(res%model_parameters%ml_only_ocean) then
                  !if(t*res%model_parameters%timestep < res%model_parameters%timestep_slab*2) then 
                  !   res%reservoir_special(i,1)%saved_state = test_state!res%reservoir_special(i,1)%saved_state!test_state
                  !   res%reservoir_special(i,1)%feedback = test_feedback!res%reservoir_special(i,1)%feedback!test_feedback
                  !endif 
                  call predict_slab_ml(res%reservoir_special(i,1),res%model_parameters,res%grid_special(i,1),res%reservoir_special(i,1)%saved_state)
                else
                  call predict_slab(res%reservoir_special(i,1),res%model_parameters,res%grid_special(i,1),res%reservoir_special(i,1)%saved_state,res%reservoir_special(i,1)%local_model)
                endif 
              endif 
           endif 
        enddo

  
        if(res%model_parameters%slab_ocean_model_bool) then !if(mod(t*res%model_parameters%timestep,res%model_parameters%timestep_slab) == 0) then
           slab_model = .True.
        else
           slab_model = .False.
        endif

        if(mpi_res%is_root) print *, 'sending data and writing predictions','prediction_num prediction_num',prediction_num,'time',t

        call sendrecievegrid(res,t,slab_model)

        if(res%model_parameters%run_speedy .eqv. .False.) then
          exit
        endif
      enddo
  enddo 

  call clean_up_speedy()

  call MPI_Barrier(mpi_res%mpi_world, mpi_res%ierr)

  enddo 
  call mpi_finalize(mpi_res%ierr)

  if(res%model_parameters%irank == 0) then
     print *, 'program finished correctly'
  endif   

end program

