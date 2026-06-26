# Install script for directory: /Users/arpan/Library/pyscf/pyscf/lib/build/deps/src/libxcfun/src/functionals

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/Users/arpan/Library/pyscf/pyscf/lib/deps")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "RELEASE")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/opt/homebrew/opt/llvm/bin/llvm-objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/XCFun/functionals" TYPE FILE FILES "/Users/arpan/Library/pyscf/pyscf/lib/build/deps/src/libxcfun/src/functionals/b97c.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/XCFun/functionals" TYPE FILE FILES "/Users/arpan/Library/pyscf/pyscf/lib/build/deps/src/libxcfun/src/functionals/b97x.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/XCFun/functionals" TYPE FILE FILES "/Users/arpan/Library/pyscf/pyscf/lib/build/deps/src/libxcfun/src/functionals/b97xc.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/XCFun/functionals" TYPE FILE FILES "/Users/arpan/Library/pyscf/pyscf/lib/build/deps/src/libxcfun/src/functionals/constants.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/XCFun/functionals" TYPE FILE FILES "/Users/arpan/Library/pyscf/pyscf/lib/build/deps/src/libxcfun/src/functionals/list_of_functionals.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/XCFun/functionals" TYPE FILE FILES "/Users/arpan/Library/pyscf/pyscf/lib/build/deps/src/libxcfun/src/functionals/m0xy_fun.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/XCFun/functionals" TYPE FILE FILES "/Users/arpan/Library/pyscf/pyscf/lib/build/deps/src/libxcfun/src/functionals/pbec_eps.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/XCFun/functionals" TYPE FILE FILES "/Users/arpan/Library/pyscf/pyscf/lib/build/deps/src/libxcfun/src/functionals/pbex.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/XCFun/functionals" TYPE FILE FILES "/Users/arpan/Library/pyscf/pyscf/lib/build/deps/src/libxcfun/src/functionals/pw92eps.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/XCFun/functionals" TYPE FILE FILES "/Users/arpan/Library/pyscf/pyscf/lib/build/deps/src/libxcfun/src/functionals/pw9xx.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/XCFun/functionals" TYPE FILE FILES "/Users/arpan/Library/pyscf/pyscf/lib/build/deps/src/libxcfun/src/functionals/pz81c.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/XCFun/functionals" TYPE FILE FILES "/Users/arpan/Library/pyscf/pyscf/lib/build/deps/src/libxcfun/src/functionals/revtpssc_eps.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/XCFun/functionals" TYPE FILE FILES "/Users/arpan/Library/pyscf/pyscf/lib/build/deps/src/libxcfun/src/functionals/revtpssx_eps.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/XCFun/functionals" TYPE FILE FILES "/Users/arpan/Library/pyscf/pyscf/lib/build/deps/src/libxcfun/src/functionals/slater.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/XCFun/functionals" TYPE FILE FILES "/Users/arpan/Library/pyscf/pyscf/lib/build/deps/src/libxcfun/src/functionals/SCAN_like_eps.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/XCFun/functionals" TYPE FILE FILES "/Users/arpan/Library/pyscf/pyscf/lib/build/deps/src/libxcfun/src/functionals/tpssc_eps.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/XCFun/functionals" TYPE FILE FILES "/Users/arpan/Library/pyscf/pyscf/lib/build/deps/src/libxcfun/src/functionals/tpssx_eps.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/XCFun/functionals" TYPE FILE FILES "/Users/arpan/Library/pyscf/pyscf/lib/build/deps/src/libxcfun/src/functionals/vwn.hpp")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/Users/arpan/Library/pyscf/pyscf/lib/build/deps/src/libxcfun-build/src/functionals/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
