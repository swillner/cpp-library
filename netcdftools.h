/*
  Copyright (C) 2019 Sven Willner <sven.willner@gmail.com>

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU Affero General Public License as published
  by the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU Affero General Public License for more details.

  You should have received a copy of the GNU Affero General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef NETCDFTOOLS_H
#define NETCDFTOOLS_H

#if defined(__CUDACC__)
#pragma push
#pragma diag_suppress = useless_type_qualifier_on_return_type
#elif defined(_MSC_VER) || defined(__INTEL_COMPILER)
#pragma warning push
#pragma warning disable : 858
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif

// IWYU pragma: begin_exports
#include "ncCompoundType.h"
#include "ncDim.h"
#include "ncException.h"
#include "ncFile.h"
#include "ncGroup.h"
#include "ncGroupAtt.h"
#include "ncType.h"
#include "ncVar.h"
#include "ncVarAtt.h"
#include "netcdf"
// IWYU pragma: end_exports

#if defined(__CUDACC__)
#pragma pop
#elif defined(_MSC_VER) || defined(__INTEL_COMPILER)
#pragma warning pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace netCDF {

inline bool check_dimensions(const netCDF::NcVar& var, const std::vector<std::string>& names) {
    const auto& dims = var.getDims();
    if (dims.size() != names.size()) {
        return false;
    }
    for (std::size_t i = 0; i < names.size(); ++i) {
        if (!names[i].empty() && dims[i].getName() != names[i]) {
            return false;
        }
    }
    return true;
}

inline void check_file_exists(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile.good()) {
        throw std::runtime_error(filename + " not found");
    }
}

template<typename Function>
inline void for_type(netCDF::NcType::ncType t, Function&& f) {
    switch (t) {
        case netCDF::NcType::nc_DOUBLE: {
            double type = 0;
            f(type);
        } break;
        case netCDF::NcType::nc_FLOAT: {
            float type = 0;
            f(type);
        } break;
        case netCDF::NcType::nc_USHORT: {
            int16_t type = 0;
            f(type);
        } break;
        case netCDF::NcType::nc_INT: {
            int type = 0;
            f(type);
        } break;
        case netCDF::NcType::nc_BYTE: {
            signed char type = 0;
            f(type);
        } break;
        default:
            throw std::runtime_error("Variable type not supported");
    }
}

}  // namespace netCDF

#endif
