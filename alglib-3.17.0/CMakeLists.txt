set(TARGET alglib)

# List of CPP (source) library files.
set(${TARGET}_SRC
        alglibinternal.cpp
        alglibmisc.cpp
        ap.cpp
        dataanalysis.cpp
        diffequations.cpp
        fasttransforms.cpp
        integration.cpp
        interpolation.cpp
        linalg.cpp
        optimization.cpp
        solvers.cpp
        specialfunctions.cpp
        statistics.cpp
        )

# List of HPP (header) library files.
set(${TARGET}_HDR
        alglibinternal.h
        alglibmisc.h
        ap.h
        dataanalysis.h
        diffequations.h
        fasttransforms.h
        integration.h
        interpolation.h
        linalg.h
        optimization.h
        solvers.h
        specialfunctions.h
        statistics.h
        stdafx.h
        )

# Adds a library target called ${LIBRARY_TARGET_NAME} to be built from the
# source and header files listed in the command invocation.
add_library(${TARGET} ${${TARGET}_SRC} ${${LIBRARY_TARGET_NAME}_HDR})

# Specify installation targets, typology and destination folders.
install(TARGETS ${TARGET}
        EXPORT  ${PROJECT_NAME}
        LIBRARY       DESTINATION "${CMAKE_INSTALL_LIBDIR}"                            COMPONENT shlib
        ARCHIVE       DESTINATION "${CMAKE_INSTALL_LIBDIR}"                            COMPONENT lib
        RUNTIME       DESTINATION "${CMAKE_INSTALL_BINDIR}"                            COMPONENT bin
        PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" COMPONENT dev)

message(STATUS "Created target ${TARGET} for export ${PROJECT_NAME}.")