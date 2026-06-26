#----------------------------------------------------------------
# Generated CMake target import file for configuration "RELEASE".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Libxc::xc" for configuration "RELEASE"
set_property(TARGET Libxc::xc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(Libxc::xc PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libxc.15.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libxc.15.dylib"
  )

list(APPEND _cmake_import_check_targets Libxc::xc )
list(APPEND _cmake_import_check_files_for_Libxc::xc "${_IMPORT_PREFIX}/lib/libxc.15.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
