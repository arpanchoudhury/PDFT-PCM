# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/Users/arpan/Library/pyscf/pyscf/lib/build/deps/src/libxc")
  file(MAKE_DIRECTORY "/Users/arpan/Library/pyscf/pyscf/lib/build/deps/src/libxc")
endif()
file(MAKE_DIRECTORY
  "/Users/arpan/Library/pyscf/pyscf/lib/build/deps/src/libxc-build"
  "/Users/arpan/Library/pyscf/pyscf/lib/deps"
  "/Users/arpan/Library/pyscf/pyscf/lib/build/deps/tmp"
  "/Users/arpan/Library/pyscf/pyscf/lib/build/deps/src/libxc-stamp"
  "/Users/arpan/Library/pyscf/pyscf/lib/build/deps/src"
  "/Users/arpan/Library/pyscf/pyscf/lib/build/deps/src/libxc-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/Users/arpan/Library/pyscf/pyscf/lib/build/deps/src/libxc-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/Users/arpan/Library/pyscf/pyscf/lib/build/deps/src/libxc-stamp${cfgdir}") # cfgdir has leading slash
endif()
