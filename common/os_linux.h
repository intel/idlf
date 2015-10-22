/*
Copyright (c) 2015, Intel Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
* Neither the name of Intel Corporation nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once

#include <dlfcn.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <string>
#include <sstream>
#include <iomanip>
#include <vector>

#if defined __GNUC__ && (__GNUC__*10000+__GNUC_MINOR__*100+__GNUC_PATCHLEVEL__)<40900
// GCC prior 4.9 has an invalid regex support in std::
// adding simple stub that emulates minimal functionality using POSIX <regex.h>

#   include <iterator>
#   include <stdexcept>
#   include <regex.h>
    namespace std {
        namespace regex_constants {
            enum error_type { error_syntax };
        }
        class regex_error : public runtime_error {
        public:
            regex_error(regex_constants::error_type error) : runtime_error("regex syntex error") {}
        }; // class regex_error

        struct regex {
            regex_t compiled_;
            regex(const char *pattern) {
                if(int status = regcomp(&compiled_, pattern, REG_EXTENDED))
                    throw regex_error(regex_constants::error_syntax);
            }
            ~regex() {
                regfree(&compiled_);
            }
        }; // struct regex

        struct cmatch {
            std::vector<std::string> array_;
            std::string& operator[](size_t at) {
                return array_[at];
            }
        }; // struct cmatch

        inline bool regex_match(const char *str, const regex &rgx) {
            return !regexec(&rgx.compiled_, str, 0, 0, 0);
        }

        inline bool regex_match(const char *str, cmatch &result, const regex &rgx) {
            const size_t C_max_match_count = 16;
            regmatch_t match_array[C_max_match_count];
            if(regexec(&rgx.compiled_, str, C_max_match_count, match_array, 0)) return false;
            else {
                result.array_.clear();
                for(size_t at=0; match_array[at].rm_so!=-1; ++at)
                result.array_.push_back(std::string(str+match_array[at].rm_so, match_array[at].rm_eo - match_array[at].rm_so));
                return true;
            }
        }

    } // std

#else
#    include <regex>
#endif
namespace {
    bool is_regular_file(std::string& dirname,struct dirent* folder_entry)
    {
        switch(folder_entry->d_type)
        {
            case DT_REG:
                return true;
            case DT_UNKNOWN:
            {
                struct stat file_stats;
                std::string absolute_file_path = dirname + "/" + folder_entry->d_name;
                // If for some reason stat was successful then skip this file
                if((stat(absolute_file_path.c_str(),&file_stats) == 0) && (S_ISREG(file_stats.st_mode))) {
                    return true;
                }
                break;
            }
            default:
                break;
        }
        return false;
    }

    std::string get_timestamp() {
        std::stringstream html_filename;
        std::time_t now = std::time(nullptr);
        std::tm * ptm = std::localtime(&now);
        html_filename.fill('0');
        html_filename << ptm->tm_year+1900
                      << std::setw(2) << ptm->tm_mon+1
                      << std::setw(2) << ptm->tm_mday
                      << std::setw(2) << ptm->tm_hour
                      << std::setw(2) << ptm->tm_min
                      << std::setw(2) << ptm->tm_sec;
        return html_filename.str();
    }

    const std::string show_HTML_command("lynx ");
    const std::string dynamic_library_extension(".so");
}
