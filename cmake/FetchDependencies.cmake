include(FetchContent)

####################################### Fetch ThreadPool #######################################
FetchContent_Declare(
  thread_pool
  GIT_REPOSITORY https://github.com/Maged152/ThreadPool.git
  GIT_TAG v1.1.0 
)

set(ThreadPool_BUILD_EXAMPLES OFF CACHE BOOL "Disable examples" FORCE)
set(ThreadPool_BUILD_Doc OFF CACHE BOOL "Disable doc" FORCE)

FetchContent_MakeAvailable(thread_pool)

####################################### Fetch googletest #######################################
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.15.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)