// Copyright (C) 2003-2008 Anders Logg and Jim Tilander.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug, 2007, 2009.
//
// First added:  2003-03-13
// Last changed: 2009-05-11

#ifndef __LOG_H
#define __LOG_H

#include <string>
#include <dolfin/common/types.h>

namespace dolfin
{

  class Variable;
  class NewParameters;

  /// The DOLFIN log system provides the following set of functions for
  /// uniform handling of log messages, warnings and errors. In addition,
  /// macros are provided for debug messages and assertions.
  ///
  /// Only messages with a debug level higher than or equal to the global
  /// debug level are printed (the default being zero). The global debug
  /// level may be controlled by
  ///
  ///    set("debug level", debug_level);
  ///
  /// where debug_level is the desired debug level.
  ///
  /// The output destination can be controlled by
  ///
  ///    set("output destination", destination);
  ///
  /// where destination is one of "terminal" (default) or "silent". Setting
  /// the output destination to "silent" means no messages will be printed.

  /// Print message
  void info(std::string msg, ...);

  /// Print message at given debug level
  void info(int debug_level, std::string msg, ...);

  /// Print variable (using output of str() method)
  void info(const Variable& variable);

  /// Print variable (using output of str() method)
  void info(const NewParameters& parameters);

  /// Print message to stream
  void info_stream(std::ostream& out, std::string msg);

  /// Print underlined message
  void info_underline(std:: string msg, ...);

  /// Print warning
  void warning(std::string msg, ...);

  /// Print error message and throw an exception
  void error(std::string msg, ...);

  /// Begin task (increase indentation level)
  void begin(std::string msg, ...);

  /// Begin task (increase indentation level)
  void begin(int debug_level, std::string msg, ...);

  /// End task (decrease indentation level)
  void end();

  /// Indent string
  std::string indent(std::string s);

  /// Print summary of timings and tasks, optionally clearing stored timings
  void summary(bool reset=false);

  /// Return timing (average) for given task, optionally clearing timing for task
  double timing(std::string task, bool reset=false);

  // Helper function for dolfin_debug macro
  void __debug(std::string file, unsigned long line, std::string function, std::string format, ...);

  // Helper function for dolfin_assert macro
  void __dolfin_assert(std::string file, unsigned long line, std::string function, std::string format, ...);

}

// Debug macros (with varying number of arguments)
#define dolfin_debug(msg)              do { dolfin::__debug(__FILE__, __LINE__, __FUNCTION__, msg); } while (false)
#define dolfin_debug1(msg, a0)         do { dolfin::__debug(__FILE__, __LINE__, __FUNCTION__, msg, a0); } while (false)
#define dolfin_debug2(msg, a0, a1)     do { dolfin::__debug(__FILE__, __LINE__, __FUNCTION__, msg, a0, a1); } while (false)
#define dolfin_debug3(msg, a0, a1, a2) do { dolfin::__debug(__FILE__, __LINE__, __FUNCTION__, msg, a0, a1, a2); } while (false)

// Assertion, only active if DEBUG is defined
#ifdef DEBUG
#define dolfin_assert(check) do { if ( !(check) ) { dolfin::__dolfin_assert(__FILE__, __LINE__, __FUNCTION__, "(" #check ")"); } } while (false)
#else
#define dolfin_assert(check)
#endif

// Not implemented error, reporting function name and line number
#define dolfin_not_implemented() do { dolfin::__debug(__FILE__, __LINE__, __FUNCTION__, "Sorry, this function has not been implemented."); error("Not implemented"); } while (false)

#endif
