# keep_case.cmake -- copy a driver with only one `case` of its order switch.
#
#   cmake -DIN=<driver.cpp> -DOUT=<copy.cpp> -DORDER=<n> -P keep_case.cmake
#
# Deletes every line that starts with optional blanks and `case <number>:`,
# except `case <ORDER>:`, so the binary instantiates the kernels of one order
# only.  Fails if the driver has no `case <ORDER>:`.

foreach(var IN ITEMS IN OUT ORDER)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "keep_case.cmake: -D${var}=... missing")
    endif()
endforeach()

file(READ "${IN}" text)
# Mark the line to keep, drop every other case line, unmark.  `@KEEP@`
# stands between the blanks and `case`, so the drop pattern skips that line.
string(REGEX REPLACE "\n([ ]*)case ${ORDER}:" "\n\\1@KEEP@case ${ORDER}:" text "${text}")
if(NOT text MATCHES "@KEEP@case ${ORDER}:")
    message(FATAL_ERROR "${IN} has no case ${ORDER}")
endif()
string(REGEX REPLACE "\n[ ]*case [0-9]+:[^\n]*" "" text "${text}")
string(REPLACE "@KEEP@" "" text "${text}")
file(WRITE "${OUT}.tmp" "${text}")
file(COPY_FILE "${OUT}.tmp" "${OUT}" ONLY_IF_DIFFERENT)
file(REMOVE "${OUT}.tmp")
