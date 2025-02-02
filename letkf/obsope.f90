PROGRAM obsope
!=======================================================================
!
! [PURPOSE:] Main program of observation operator
!
! [HISTORY:]
!   04/03/2013 Takemasa Miyoshi  created
!
!=======================================================================
  USE common
  USE common_speedy
  USE common_obs_speedy

  IMPLICIT NONE
  CHARACTER(9) :: obsinfile='obsin.dat'    !IN
  CHARACTER(8) :: guesfile='gues.grd'      !IN
  CHARACTER(10) :: obsoutfile='obsout.dat' !OUT
  REAL(r_size),ALLOCATABLE :: elem(:)
  REAL(r_size),ALLOCATABLE :: rlon(:)
  REAL(r_size),ALLOCATABLE :: rlat(:)
  REAL(r_size),ALLOCATABLE :: rlev(:)
  REAL(r_size),ALLOCATABLE :: odat(:)
  REAL(r_size),ALLOCATABLE :: oerr(:)
  REAL(r_size),ALLOCATABLE :: ohx(:)
  INTEGER,ALLOCATABLE :: oqc(:)
  REAL(r_size),ALLOCATABLE :: v3d(:,:,:,:)
  REAL(r_size),ALLOCATABLE :: v2d(:,:,:)
  REAL(r_size),ALLOCATABLE :: p_full(:,:,:)
  REAL(r_size),PARAMETER :: threshold_dz=1000.0d0
  REAL(r_size) :: dz,tg,qg
  INTEGER :: nobs
  REAL(r_size) :: ri,rj,rk
  INTEGER :: n
  INTEGER :: rejects

  CALL set_common_speedy

  CALL get_nobs(obsinfile,6,nobs)
  ALLOCATE( elem(nobs) )
  ALLOCATE( rlon(nobs) )
  ALLOCATE( rlat(nobs) )
  ALLOCATE( rlev(nobs) )
  ALLOCATE( odat(nobs) )
  ALLOCATE( oerr(nobs) )
  ALLOCATE( ohx(nobs) )
  ALLOCATE( oqc(nobs) )
  CALL read_obs(obsinfile,nobs,elem,rlon,rlat,rlev,odat,oerr)
  ALLOCATE( v3d(nlon,nlat,nlev,nv3d) )
  ALLOCATE( v2d(nlon,nlat,nv2d) )
  CALL read_grd(guesfile,v3d,v2d)
  ALLOCATE( p_full(nlon,nlat,nlev) )
  CALL calc_pfull(nlon,nlat,v2d(:,:,iv2d_ps),p_full)

  ! test dylan call dz and call rlev(n)
  print *, 'dylan print rlev(n), dz before do loop', rlev(n), dz

  ohx=0.0d0
  oqc=0
  rejects = 0
  DO n=1,nobs
    print *, 'dylan before call phys2ijk rlev(n), dz', rlev(n), dz
    CALL phys2ijk(p_full,elem(n),rlon(n),rlat(n),rlev(n),ri,rj,rk)
    IF(CEILING(ri) < 2 .OR. nlon+1 < CEILING(ri)) THEN
      WRITE(6,'(A)') '* X-coordinate out of range'
      WRITE(6,'(A,F6.2,A,F6.2)') '*   ri=',ri,', rlon=',rlon(n)
      rejects = rejects + 1
      CYCLE
    END IF
    IF(CEILING(rj) < 2 .OR. nlat < CEILING(rj)) THEN
      WRITE(6,'(A)') '* Y-coordinate out of range'
      WRITE(6,'(A,F6.2,A,F6.2)') '*   rj=',rj,', rlat=',rlat(n)
      rejects = rejects + 1
      CYCLE
    END IF
    IF(CEILING(rk) > nlev) THEN
      print *, 'dylan ceiling dz ', dz
      
      CALL itpl_2d(phi0,ri,rj,dz)
      WRITE(6,'(A)') '* Z-coordinate out of range'
      WRITE(6,'(A,F6.2,A,F10.2,A,F6.2,A,F6.2,A,F10.2)') &
       & '*   rk=',rk,', rlev=',rlev(n),&
       & ', (lon,lat)=(',rlon(n),',',rlat(n),'), phi0=',dz 
       rejects = rejects + 1
      CYCLE
    END IF
    IF(CEILING(rk) < 2 .AND. NINT(elem(n)) /= id_ps_obs) THEN
      IF(NINT(elem(n)) == id_u_obs .OR. NINT(elem(n)) == id_v_obs) THEN
        rk = 1.00001d0
      ELSE
        CALL itpl_2d(phi0,ri,rj,dz)
        WRITE(6,'(A)') '* Z-coordinate out of range'
        WRITE(6,'(A,F6.2,A,F10.2,A,F6.2,A,F6.2,A,F10.2)') &
         & '*   rk=',rk,', rlev=',rlev(n),&
         & ', (lon,lat)=(',rlon(n),',',rlat(n),'), phi0=',dz
        CYCLE
        rejects = rejects + 1
      END IF
    END IF
    IF(NINT(elem(n)) == id_ps_obs .AND. odat(n) < -100.0d0) THEN
      rejects = rejects + 1
      CYCLE
    END IF

    !NOTE This was commented out before - now dylan 6/21 commented it out again
    !to test if it works.. 
    ! NOW BACK ON dylan 7/2/24 
    IF(NINT(elem(n)) /= id_ps_obs) THEN
      print *, 'ELEMENT NUMBER NOT ID PS OBS'  
      print *,'Dylan here. ', ' dz ', dz
      print *,'rlev(n) ', rlev(n)
    END IF

   IF(NINT(elem(n)) == id_ps_obs) THEN
     print *, 'WHERE elem(n) == id_ps_obs'  
     print *,'BEFORE ITPL_2D, rlev(n), dz = ',rlev(n),dz !calling phi0 print
     ! is a disaster
     CALL itpl_2d(phi0,ri,rj,dz)
     !print '(A,F10.2,F10.2)','tmplev(n),dz',rlev(n),dz !ORIGINAL LINE HERE 
     
     print *,'AFTER ITPL_2D, rlev(n), dz = ',rlev(n),dz
     dz = dz - rlev(n)
     print *, 'now its dz - rlev(n) ', dz
     IF(ABS(dz) < threshold_dz) THEN ! pressure adjustment threshold
       CALL itpl_2d(v3d(:,:,1,3),ri,rj,tg) !Temperature
       CALL itpl_2d(v3d(:,:,1,4),ri,rj,qg) !Specific humidity 
       CALL prsadj(odat(n),dz,tg,qg)
     ELSE
       PRINT '(A)','PS obs vertical adjustment beyond threshold'
       PRINT '(A,F10.2,A,F6.2,A,F6.2,A)',&
         & '  dz=',dz,', (lon,lat)=(',rlon(n),',',rlat(n),')'
       rejects = rejects + 1
       CYCLE
     END IF
   END IF

    !
    ! observational operator
    !
    CALL Trans_XtoY(elem(n),ri,rj,rk,v3d,v2d,p_full,ohx(n))
    oqc(n) = 1
  END DO

  CALL write_obs2(obsoutfile,nobs,elem,rlon,rlat,rlev,odat,oerr,ohx,oqc)

  print '(A,I0,A,I0)','Total Number of Possible Obs',nobs,'Rejected number of obs',rejects
  DEALLOCATE( elem,rlon,rlat,rlev,odat,oerr,ohx,oqc,v3d,v2d,p_full )

END PROGRAM obsope
