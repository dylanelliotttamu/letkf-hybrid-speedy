module troy_stuff

use mod_utilities, only : dp

real(kind=dp), parameter :: e_constant = 2.7182818284590452353602874_dp

contains 

  SUBROUTINE read_era5_netcdf(date,var3d,var2d)
    USE common
    USE common_speedy
    USE common_obs_speedy

    use datetime_module, only : timedelta, datetime
    use mod_io, only : read_netcdf_4d, read_netcdf_3d

    implicit none

    character(len=*), intent(inout) :: date

    real(r_size), intent(inout) :: var3d(:,:,:,:), var2d(:,:,:)

    real(r_size), allocatable :: temp4d(:,:,:,:), temp3d(:,:,:), temp2d(:,:,:)

    type (datetime) :: start_of_year, current_date
    type (timedelta) :: t

    integer :: year_i, month_i, day_i, hour_i, hour_into_year

    character(len=3) :: file_end='.nc'
    character(len=7) :: file_begin = 'era_5_y'
    character(len=28) :: regrid_mpi = '_regridded_mpi_fixed_var_gcc'
    character(len=2) :: mid_file='_y'
    character(len=4) :: year
    character(len=:), allocatable :: file_path
    character(len=:), allocatable :: regrid_file_name
    character(len=:), allocatable :: format_month
    character(len=:), allocatable :: month
    character(len=:), allocatable :: precip_file_name

    read(date(1:4),'(I4)') year_i
    read(date(5:6),'(I2)') month_i
    read(date(7:8),'(I2)') day_i
    read(date(9:10),'(I2)') hour_i

    current_date = datetime(year_i,month_i,day_i,hour_i)
    start_of_year = datetime(current_date%getYear(), 1, 1, 0)

    t = current_date - start_of_year

    hour_into_year = int(t%total_seconds()/3600) + 1
    print *, 'hour into year',hour_into_year

    file_path = '/scratch/user/troyarcomano/ERA_5/'//date(1:4)//'/'
    regrid_file_name = file_path//file_begin//date(1:4)//regrid_mpi//file_end

    precip_file_name = file_path//'era_5_y'//date(1:4)//'_precip_regridded_mpi_fixed_var_gcc.nc'

    call read_netcdf_4d('Temperature',regrid_file_name,temp4d,hour_into_year)
    var3d(:,:,:,3) = temp4d(:,:,8:1:-1,1)
    deallocate(temp4d)

    call read_netcdf_4d('U-wind',regrid_file_name,temp4d,hour_into_year)
    var3d(:,:,:,1) = temp4d(:,:,8:1:-1,1)
    deallocate(temp4d)

    call read_netcdf_4d('V-wind',regrid_file_name,temp4d,hour_into_year)
    var3d(:,:,:,2) = temp4d(:,:,8:1:-1,1)
    
    deallocate(temp4d)

    call read_netcdf_4d('Specific_Humidity',regrid_file_name,temp4d,hour_into_year)
    var3d(:,:,:,4) = temp4d(:,:,8:1:-1,1)
    deallocate(temp4d)

    call read_netcdf_3d('logp',regrid_file_name,temp3d)
    var2d(:,:,1) = e_constant**temp3d(:,:,hour_into_year) * 100000.0_dp
    deallocate(temp3d)
 
    call read_netcdf_3d('tp',precip_file_name,temp3d)
    var2d(:,:,2) = temp3d(:,:,hour_into_year)* 6 * 1000.0_dp!NOTE need to change this 
    deallocate(temp3d) 
    !nlon,nlat,nlev,nv3d
  end subroutine read_era5_netcdf

  subroutine flip(original_list,flipped_list)
     real(kind=dp), intent(in) :: original_list(:)
     real(kind=dp), intent(inout) :: flipped_list(:)

     integer :: i, list_size

     list_size = size(original_list)

     flipped_list = original_list(list_size:1:-1)

     return
   end subroutine
end module 
