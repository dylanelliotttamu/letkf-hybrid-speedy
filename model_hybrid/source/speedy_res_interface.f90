module speedy_res_interface
    
   use mod_utilities, only : dp, speedy_data_type, era_data_type, state_vector_type, &
                             reservoir_type, grid_type, model_parameters_type
   use mod_atparam, only : ix,il,kx
   use mod_calendar, only : calendar, initialize_calendar

   implicit none 

   integer, parameter :: numoftimestep=17, stride=1, & !13140*6 !18300*4 this is for 50 time step days 
                         vartime=numoftimestep/stride,   &
                         numofspeedyvars=4, numoflevels=8 

   type(state_vector_type) :: internal_state_vector

 contains 
   subroutine startspeedy(model_parameters,grid,runspeedy)
      use mpires, only : mpi_res 
      use mod_io, only : write_netcdf_speedy_full, read_era_data_parallel
      use resdomain, only : initializedomain

      type(model_parameters_type), intent(in) :: model_parameters
      type(grid_type), intent(inout)          :: grid
      
      logical, intent(in) :: runspeedy

      integer, parameter  :: root=0

      integer             :: speedydays
      integer             :: vert_level
      
      call initializedomain(mpi_res%numprocs,mpi_res%proc_num,model_parameters%overlap,grid%num_vert_levels,vert_level,grid%vert_overlap,grid)
      
      call initialize_calendar(calendar,1981,1,1,0)!call initialize_calendar(calendar,1982,1,1,0)
   end subroutine startspeedy


  subroutine read_era(reservoir,grid,model_parameters,start_year,end_year,era_data,timestep_arg)
     use mpires, only : mpi_res
     use mod_io, only : read_era_data_parallel,read_3d_file_parallel, read_era_data_parallel_old, &
                        read_3d_file_parallel_res
     use mod_calendar, only : numof_hours

     type(reservoir_type), intent(inout)     :: reservoir
     type(grid_type), intent(inout)          :: grid
     type(model_parameters_type), intent(in) :: model_parameters

     integer, intent(in)                     :: start_year, end_year

     type(era_data_type), intent(inout)      :: era_data

     integer, intent(in), optional           :: timestep_arg

     integer :: year_i, month_i,start_month,end_month
     integer :: numofhours, hour_counter, temp_length, temp_file_length
     integer :: start_index, timestep

     type(era_data_type) :: era_data_temp

     character(len=3) :: file_end='.nc'
     character(len=7) :: file_begin = 'era_5_y'
     !character(len=23) :: spectral_regrid_file  = '_regridded_spectral_mpi'
     !character(len=14) :: regrid_mpi = '_regridded_mpi'
     !character(len=18) :: regrid_mpi = '_regridded_mpi_new'
     !character(len=24) :: regrid_mpi = '_regridded_mpi_fixed_var'
     character(len=28) :: regrid_mpi = '_regridded_mpi_fixed_var_gcc'
     character(len=2) :: mid_file='_y'
     character(len=1) :: month_1
     character(len=2) :: month_2
     character(len=4) :: year
     character(len=:), allocatable :: file_path
     character(len=:), allocatable :: regrid_file_name
     character(len=:), allocatable :: spectral_regrid_file_name
     character(len=:), allocatable :: format_month
     character(len=:), allocatable :: month
     character(len=:), allocatable :: tisr_file
     character(len=:), allocatable :: sst_file
     character(len=:), allocatable :: sst_climo_file
     character(len=:), allocatable :: precip_file

     !-----------Troy stuff ---------------!

     if(present(timestep_arg)) then 
       timestep = timestep_arg
     else 
       timestep = model_parameters%timestep
     endif 

     start_month = 1
     end_month = 12

     call numof_hours(start_year,end_year,numofhours)


     allocate(era_data%eravariables(numofspeedyvars,grid%inputxchunk,grid%inputychunk,grid%inputzchunk,numofhours))
     allocate(era_data%era_logp(grid%inputxchunk,grid%inputychunk,numofhours))


     if(reservoir%tisr_input_bool) then
       allocate(era_data%era_tisr(grid%inputxchunk,grid%inputychunk,numofhours))
     endif

     if(reservoir%sst_bool) then
       allocate(era_data%era_sst(grid%inputxchunk,grid%inputychunk,numofhours+1))
     endif

     if(reservoir%sst_climo_bool) then
       !allocate(era_data%era_sst_climo(grid%resxchunk,grid%resychunk,numofhours))
       allocate(era_data%era_sst_climo(grid%inputxchunk,grid%inputychunk,numofhours+1))
     endif
    
     if(reservoir%precip_bool) then
       allocate(era_data%era_precip(grid%inputxchunk,grid%inputychunk,numofhours))
     endif


     !print *, numofhours
     hour_counter = 1 
    
     print *, 'herhe' 
     print *, 'start_year,end_year',start_year,end_year
     do year_i=start_year,end_year
         write(year,'(I4)') year_i

         if(allocated(era_data_temp%era_sst_climo)) print *, 'top loop temp climo allocated',year_i
         file_path = '/skydata2/troyarcomano/ERA_5/'//year//'/'
         regrid_file_name = file_path//file_begin//year//regrid_mpi//file_end

         print *, 'regrid_file_name',regrid_file_name 
         print *, 'callimng read_era_data_parallel'
         if(allocated(era_data_temp%era_sst_climo)) print *, 'before read_era_data_parallel loop temp climo allocated',year_i
         call read_era_data_parallel(regrid_file_name,model_parameters,mpi_res,grid,era_data_temp,1,1)
         if(allocated(era_data_temp%era_sst_climo)) print *, 'after read_era_data_parallel loop temp climo allocated',year_i
         !call read_era_data_parallel_old(regrid_file_name,mpi_res,grid,era_data_temp)
   
         if(reservoir%assigned_region == 954) print *, 'era_data_temp%eravariables(4,2,2,:,1)', era_data_temp%eravariables(4,2,2,:,1)

         temp_length = size(era_data_temp%eravariables,5)!/timestep
         !temp_file_length = temp_length * timestep

         !print *, 'shape(era_data%eravariables(:,:,:,:,hour_counter:temp_length+hour_counter-1)', shape(era_data%eravariables(:,:,:,:,hour_counter:temp_length+hour_counter-1))
         !print *, 'shape( era_data_temp%eravariables(:,:,:,:,start_index:temp_file_length:model_parameters%timestep))',shape(era_data_temp%eravariables(:,:,:,:,start_index:temp_file_length:timestep))
         !print *, 'grid%input_zstart:grid%input_zend',grid%input_zstart,grid%input_zend
         era_data%eravariables(:,:,:,:,hour_counter:temp_length+hour_counter-1) = era_data_temp%eravariables(:,:,:,grid%input_zstart:grid%input_zend,:)!,start_index:temp_file_length:timestep)
         era_data%era_logp(:,:,hour_counter:temp_length+hour_counter-1) = era_data_temp%era_logp(:,:,:)!,start_index:temp_file_length:timestep)

         if(allocated(era_data_temp%era_sst_climo)) print *, 'before tisr loop temp climo allocated',year_i

         !if(reservoir%assigned_region == 954) print *, 'read parallel era_data%eravariables(4,2,2,:,1)', era_data%eravariables(4,2,2,:,1)
         if(reservoir%tisr_input_bool) then
           !tisr_file = file_path//'toa_incident_solar_radiation_'//year//'_regridded_mpi_fixed_var.nc'
           !tisr_file = file_path//'toa_incident_solar_radiation_'//year//'_regridded_mpi.nc'
           tisr_file = file_path//'toa_incident_solar_radiation_'//year//'_regridded_classic4.nc'
           if(model_parameters%irank == 0) print *, tisr_file
           call read_3d_file_parallel(tisr_file,'tisr',mpi_res,grid,era_data_temp%era_tisr,1,1)
           era_data%era_tisr(:,:,hour_counter:temp_length+hour_counter-1) = era_data_temp%era_tisr(:,:,:)!,start_index:temp_file_length:timestep)
           !print *, 'era_data%era_tisr(1,1,400:500) in speedy_res_inferface',era_data%era_tisr(1,1,400:500)
         endif
         if(allocated(era_data_temp%era_sst_climo)) print *, 'after era_tisr loop temp climo allocated',year_i
         print *, 'reservoir%precip_bool',reservoir%precip_bool,reservoir%assigned_region
         if(reservoir%precip_bool) then 
           precip_file = file_path//'era_5_y'//year//'_precip_regridded_mpi_fixed_var_gcc.nc'
           if(model_parameters%irank == 0) print *, precip_file
           call read_3d_file_parallel(precip_file,'tp',mpi_res,grid,era_data_temp%era_precip,1,1)
           !print *, 'era_data_temp%era_precip(1,1,10:12)',era_data_temp%era_precip(1,1,10:12)
           print *, 'hour_counter:temp_length+hour_counter-1',hour_counter,temp_length+hour_counter-1, shape(era_data%era_precip)
           era_data%era_precip(:,:,hour_counter:temp_length+hour_counter-1) = era_data_temp%era_precip
         endif 

         !print *, 'reservoir%sst_bool',reservoir%sst_bool,reservoir%assigned_region
         if(reservoir%sst_bool) then
           sst_file = file_path//'era_5_y'//year//'_sst_regridded_fixed_var_gcc.nc'   !'_sst_regridded_mpi_fixed_var_gcc.nc'
           if(model_parameters%irank == 0) print *, sst_file
           call read_3d_file_parallel(sst_file,'sst',mpi_res,grid,era_data_temp%era_sst,1,1)
           era_data%era_sst(:,:,hour_counter:temp_length+hour_counter-1) = era_data_temp%era_sst
         endif

         print *, 'reservoir%sst_climo_bool',reservoir%sst_climo_bool, reservoir%assigned_region
         if(reservoir%sst_climo_bool) then
           sst_climo_file = '/skydata2/troyarcomano/ERA_5/regridded_era_sst_climatology1981_1999_gcc.nc'
           if(year_i == start_year) then
              if(model_parameters%irank == 0) print *, sst_climo_file
              !call read_3d_file_parallel_res(sst_climo_file,'sst',mpi_res,grid,era_data_temp%era_sst_climo) 
              call read_3d_file_parallel(sst_climo_file,'sst',mpi_res,grid,era_data_temp%era_sst_climo,1,1)
           endif
           if(temp_length == 8784) then  
             era_data%era_sst_climo(:,:,hour_counter:hour_counter+1440 - 1) = era_data_temp%era_sst_climo(:,:,1:1440)
             era_data%era_sst_climo(:,:,hour_counter+1441:hour_counter+1440+24) = era_data_temp%era_sst_climo(:,:,1441-24:1440)
             era_data%era_sst_climo(:,:,hour_counter+1465:hour_counter+8784) = era_data_temp%era_sst_climo(:,:,1441:8760)
           else
             print *, 'shape(era_data%era_sst_climo(:,:,hour_counter:temp_length+hour_counter-1)) era_temp',shape(era_data%era_sst_climo(:,:,hour_counter:temp_length+hour_counter-1)),shape(era_data_temp%era_sst_climo)
             if(allocated(era_data_temp%era_sst_climo)) print *, 'right before'
             era_data%era_sst_climo(:,:,hour_counter:temp_length+hour_counter-1) = era_data_temp%era_sst_climo
           endif
         endif

         if(model_parameters%train_on_sst_anomalies .and. reservoir%sst_bool) then
           if(temp_length == 8784) then
             era_data%era_sst(:,:,hour_counter:hour_counter+1440 - 1) = era_data%era_sst(:,:,hour_counter:hour_counter+1440 - 1) - era_data%era_sst_climo(:,:,hour_counter:hour_counter+1440 - 1) 
             era_data%era_sst(:,:,hour_counter+1441:hour_counter+1440+24) = era_data%era_sst(:,:,hour_counter+1441:hour_counter+1440+24) - era_data%era_sst_climo(:,:,hour_counter+1441:hour_counter+1440+24) 
             era_data%era_sst(:,:,hour_counter+1465:hour_counter+8784) = era_data%era_sst(:,:,hour_counter+1465:hour_counter+8784) - era_data%era_sst_climo(:,:,hour_counter+1465:hour_counter+8784) 
           else 
             print *, 'shape,era_sst, era_sst_climo',shape(era_data%era_sst(:,:,hour_counter:temp_length+hour_counter-1)),shape(era_data%era_sst_climo(:,:,hour_counter:temp_length+hour_counter-1))
             era_data%era_sst(:,:,hour_counter:temp_length+hour_counter-1) = era_data%era_sst(:,:,hour_counter:temp_length+hour_counter-1) - era_data%era_sst_climo(:,:,hour_counter:temp_length+hour_counter-1)
           endif 
         endif 

         hour_counter = temp_length+hour_counter
  
         deallocate(era_data_temp%eravariables)
         deallocate(era_data_temp%era_logp)

         if(allocated(era_data_temp%era_tisr)) then
            deallocate(era_data_temp%era_tisr)
         endif

         if(allocated(era_data_temp%era_sst)) then
            deallocate(era_data_temp%era_sst)
         endif
  
         if(allocated(era_data_temp%era_precip)) then 
           deallocate(era_data_temp%era_precip)
         endif 

         if(allocated(era_data_temp%era_sst_climo)) print *, 'bottom loop temp climo allocated',year_i
    enddo
    if(allocated(era_data_temp%era_sst_climo)) then
       deallocate(era_data_temp%era_sst_climo)
    endif
  end subroutine

 subroutine read_model_states(reservoir,grid,model_parameters,start_year,end_year,speedy_data,timestep_arg)
     use mpires, only : mpi_res
     use mod_io, only : read_speedy_data_parallel, read_speedy_data_parallel_old
     use mod_calendar, only : numof_hours

     type(reservoir_type), intent(inout)     :: reservoir
     type(grid_type), intent(inout)          :: grid
     type(model_parameters_type), intent(in) :: model_parameters

     integer, intent(in)                     :: start_year,end_year ! ,loop_index

     type(speedy_data_type), intent(inout)   :: speedy_data

     integer, intent(in), optional           :: timestep_arg

     integer :: year_i, month_i,start_month,end_month
     integer :: numofhours, hour_counter, temp_length, temp_file_length
     integer :: start_time, timestep

     type(speedy_data_type) :: speedy_data_temp

     character(len=3) :: file_end='.nc'
     !character(len=9) :: file_begin = 'restart_y'
     character(len=15) :: file_begin = 'restart_6hour_y'
     !character(len=13) :: file_begin = 'restart_1hr_y'
     !character(len=15) :: file_begin = 'restart_3hour_y'
     !character(len=24) :: file_begin = 'restart_1hr_more_steps_y'
     character(len=2) :: mid_file='_m'
     character(len=1) :: month_1
     character(len=2) :: month_2
     character(len=4) :: year
     character(len=:), allocatable :: file_path
     character(len=:), allocatable :: restart_file_name
     character(len=:), allocatable :: format_month
     character(len=:), allocatable :: month

     !-----------Troy stuff ---------------!

     if(present(timestep_arg)) then
       timestep = timestep_arg
     else
       timestep = model_parameters%timestep
     endif

     start_month = 1
     end_month = 12

     call numof_hours(start_year,end_year,numofhours)

     !numofhours = numofhours/res%timestep

     allocate(speedy_data%speedyvariables(numofspeedyvars,grid%resxchunk,grid%resychunk,grid%reszchunk,numofhours))
     allocate(speedy_data%speedy_logp(grid%resxchunk,grid%resychunk,numofhours))

     hour_counter = 1

     print *, 'herhe'
     !start_time = loop_index
     do year_i=start_year,end_year
         write(year,'(I4)') year_i

         file_path = '/skydata2/troyarcomano/SPEEDY_STATES/'
         !restart_file_name = file_path//file_begin//year//'_fixed_var'//file_end 
         !restart_file_name = file_path//file_begin//year//'_chunked_time'//file_end
         !restart_file_name = file_path//file_begin//year//'_new'//file_end
         !restart_file_name = file_path//file_begin//year//'_gcc'//file_end
         restart_file_name = file_path//file_begin//year//file_end

         if(model_parameters%irank == 0) print *, 'restart_file_name',restart_file_name
         print *, 'restart_file_name',restart_file_name
         call read_speedy_data_parallel(restart_file_name,mpi_res,grid,speedy_data_temp,1,1)
         !call read_speedy_data_parallel_old(restart_file_name,mpi_res,grid,speedy_data_temp)

         temp_length = size(speedy_data_temp%speedyvariables,5)!/timestep
         temp_file_length = temp_length !* timestep


         speedy_data%speedyvariables(:,:,:,:,hour_counter:temp_length+hour_counter-1) = speedy_data_temp%speedyvariables(:,:,:,grid%res_zstart:grid%res_zend,:)!start_time:temp_file_length:timestep)
         speedy_data%speedy_logp(:,:,hour_counter:temp_length+hour_counter-1) = speedy_data_temp%speedy_logp(:,:,:)!start_time:temp_file_length:timestep)

         hour_counter = temp_length+hour_counter

         deallocate(speedy_data_temp%speedyvariables)
         deallocate(speedy_data_temp%speedy_logp)

    enddo
  end subroutine

  subroutine read_letkf_analysis(reservoir,grid,model_parameters,start_year,end_year,era_data,timestep_arg)
     use mpires, only : mpi_res
     use mod_io, only : read_era_data_parallel,read_3d_file_parallel, read_era_data_parallel_old, &
                        read_3d_file_parallel_res
     use mod_calendar, only : numof_hours

     type(reservoir_type), intent(inout)     :: reservoir
     type(grid_type), intent(inout)          :: grid
     type(model_parameters_type), intent(in) :: model_parameters

     integer, intent(in)                     :: start_year, end_year

     type(era_data_type), intent(inout)      :: era_data

     integer, intent(in), optional           :: timestep_arg

     integer :: year_i, month_i,start_month,end_month
     integer :: numofhours, hour_counter, temp_length, temp_file_length
     integer :: start_index, timestep

     type(era_data_type) :: era_data_temp

     character(len=3) :: file_end='.nc'
     character(len=7) :: file_begin = 'era_5_y'
     !character(len=23) :: spectral_regrid_file  = '_regridded_spectral_mpi'
     !character(len=14) :: regrid_mpi = '_regridded_mpi'
     !character(len=18) :: regrid_mpi = '_regridded_mpi_new'
     !character(len=24) :: regrid_mpi = '_regridded_mpi_fixed_var'
     character(len=28) :: regrid_mpi = '_regridded_mpi_fixed_var_gcc'
     character(len=2) :: mid_file='_y'
     character(len=1) :: month_1
     character(len=2) :: month_2
     character(len=4) :: year
     character(len=:), allocatable :: file_path
     character(len=:), allocatable :: regrid_file_name
     character(len=:), allocatable :: spectral_regrid_file_name
     character(len=:), allocatable :: format_month
     character(len=:), allocatable :: month
     character(len=:), allocatable :: tisr_file
     character(len=:), allocatable :: sst_file
     character(len=:), allocatable :: sst_climo_file
     character(len=:), allocatable :: precip_file

     !-----------Troy stuff ---------------!
     if(present(timestep_arg)) then
       timestep = timestep_arg
     else
       timestep = model_parameters%timestep
     endif

     start_month = 1
     end_month = 12

     call numof_hours(start_year,end_year,numofhours)

     numofhours = numofhours/6
     
     !allocate(era_data%eravariables(numofspeedyvars,grid%inputxchunk,grid%inputychunk,grid%inputzchunk,numofhours))
     !allocate(era_data%era_logp(grid%inputxchunk,grid%inputychunk,numofhours))


     if(reservoir%tisr_input_bool) then
       allocate(era_data%era_tisr(grid%inputxchunk,grid%inputychunk,numofhours))
     endif

     if(reservoir%sst_bool) then
       allocate(era_data%era_sst(grid%inputxchunk,grid%inputychunk,numofhours+1))
     endif

     if(reservoir%sst_climo_bool) then
       !allocate(era_data%era_sst_climo(grid%resxchunk,grid%resychunk,numofhours))
       allocate(era_data%era_sst_climo(grid%inputxchunk,grid%inputychunk,numofhours+1))
     endif

     if(reservoir%precip_bool) then
       allocate(era_data%era_precip(grid%inputxchunk,grid%inputychunk,numofhours))
     endif


     !print *, numofhours
     hour_counter = 1

     print *, 'herhe'
     print *, 'start_year,end_year',start_year,end_year

     regrid_file_name = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/30yr_uniform_lektf_anal/uniform_analysis_81_12.nc'
     call read_era_data_parallel(regrid_file_name,model_parameters,mpi_res,grid,era_data,1,1)

     era_data%era_logp = log(era_data%era_logp/100000.0)
     do year_i=start_year,end_year
         write(year,'(I4)') year_i

         file_path = '/skydata2/troyarcomano/ERA_5/'//year//'/'
         if(reservoir%tisr_input_bool) then
           tisr_file = file_path//'toa_incident_solar_radiation_'//year//'_regridded_classic4.nc'
           if(model_parameters%irank == 0) print *, tisr_file
           call read_3d_file_parallel(tisr_file,'tisr',mpi_res,grid,era_data_temp%era_tisr,1,1)
           print *, 'here' 
           temp_length = size(era_data%era_tisr,3)
           era_data%era_tisr(:,:,hour_counter:temp_length+hour_counter-1) = era_data_temp%era_tisr(:,:,1:temp_length:timestep)!,start_index:temp_file_length:timestep)
         endif

         if(reservoir%sst_bool) then
           sst_file = file_path//'era_5_y'//year//'_sst_regridded_fixed_var_gcc.nc' !'_sst_regridded_mpi_fixed_var_gcc.nc'
           if(model_parameters%irank == 0) print *, sst_file
           call read_3d_file_parallel(sst_file,'sst',mpi_res,grid,era_data_temp%era_sst,1,1)
           era_data%era_sst(:,:,hour_counter:temp_length+hour_counter-1) = era_data_temp%era_sst
         endif

         hour_counter = temp_length+hour_counter

         if(allocated(era_data_temp%era_tisr)) then
           deallocate(era_data_temp%era_tisr)
         endif

         if(allocated(era_data_temp%era_sst)) then
            deallocate(era_data_temp%era_sst)
         endif

         if(allocated(era_data_temp%era_precip)) then
           deallocate(era_data_temp%era_precip)
         endif

         if(allocated(era_data_temp%era_sst_climo)) print *, 'bottom loop temp climo allocated',year_i
    enddo
    if(allocated(era_data_temp%era_sst_climo)) then
       deallocate(era_data_temp%era_sst_climo)
    endif
  end subroutine


  subroutine read_model_states_letkf(reservoir,grid,model_parameters,start_year,end_year,speedy_data,timestep_arg)
     use mpires, only : mpi_res
     use mod_io, only : read_speedy_data_parallel, read_speedy_data_parallel_old
     use mod_calendar, only : numof_hours

     type(reservoir_type), intent(inout)     :: reservoir
     type(grid_type), intent(inout)          :: grid
     type(model_parameters_type), intent(in) :: model_parameters

     integer, intent(in)                     :: start_year,end_year !,loop_index

     type(speedy_data_type), intent(inout)   :: speedy_data

     integer, intent(in), optional           :: timestep_arg

     integer :: year_i, month_i,start_month,end_month
     integer :: numofhours, hour_counter, temp_length, temp_file_length
     integer :: start_time, timestep

     type(speedy_data_type) :: speedy_data_temp

     character(len=3) :: file_end='.nc'
     !character(len=9) :: file_begin = 'restart_y'
     character(len=15) :: file_begin = 'restart_6hour_y'
     !character(len=13) :: file_begin = 'restart_1hr_y'
     !character(len=15) :: file_begin = 'restart_3hour_y'
     !character(len=24) :: file_begin = 'restart_1hr_more_steps_y'
     character(len=2) :: mid_file='_m'
     character(len=1) :: month_1
     character(len=2) :: month_2
     character(len=4) :: year
     character(len=:), allocatable :: file_path
     character(len=:), allocatable :: restart_file_name
     character(len=:), allocatable :: format_month
     character(len=:), allocatable :: month
 
     !-----------Troy stuff ---------------!

     if(present(timestep_arg)) then
       timestep = timestep_arg
     else
       timestep = model_parameters%timestep
     endif

     start_month = 1
     end_month = 12

     call numof_hours(start_year,end_year,numofhours)

     numofhours = numofhours

     allocate(speedy_data%speedyvariables(numofspeedyvars,grid%resxchunk,grid%resychunk,grid%reszchunk,numofhours))
     allocate(speedy_data%speedy_logp(grid%resxchunk,grid%resychunk,numofhours))

     hour_counter = 1

     print *, 'herhe'
     !start_time = loop_index
     write(year,'(I4)') year_i

     file_path = '/skydata2/dylanelliott/SPEEDY_STATES/'
        
     restart_file_name = '/skydata2/dylanelliott/SPEEDY_STATES/restart_1981_2011_uniform_letkf_analysis.nc'
     print *, 'restart_file_name',restart_file_name
     call read_speedy_data_parallel(restart_file_name,mpi_res,grid,speedy_data,1,1)
     !print *, 'before log speedy_data%speedy_logp(1,1,100)',speedy_data%speedy_logp(1,1,100)
     !speedy_data%speedy_logp = log(speedy_data%speedy_logp/100000.0)
     !print *, 'after speedy_data%speedy_logp(1,1,100)',speedy_data%speedy_logp(1,1,100)
  end subroutine

  function truncate_letkf_code_version(field_orig, trunc_twn) result(field_new)
            complex, intent(in) :: field_orig(:,:)
            integer, intent(in) :: trunc_twn
            complex, allocatable :: field_new(:,:)

            integer :: mx_lr, nx_lr, m, n

            allocate(field_new,mold=field_orig)
            mx_lr = size(field_orig,1)
            nx_lr = size(field_orig,2)

            do m = 1,mx_lr
                do n = 1,nx_lr
                    field_new(m,n) = field_orig(m,n)

                    if (m+n-2 > trunc_twn) then
                        field_new(m,n) = (0.0,0.0)
                    end if
                end do
            end do
  end function 
end module speedy_res_interface
