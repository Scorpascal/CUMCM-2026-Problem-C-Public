"""Reproduce the public derivative after providing the official input workbooks."""
import argparse
from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parent
COMMANDS_Q3=[['Q1/solve_q1.py'],['Q1/test_q1.py'],['Q2/solve_q2.py','--compute'],['Q2/report_q2.py'],['Q2/test_q2.py'],
 ['Q3/solve_q3.py','--prototype'],['Q3/solve_q3.py','--compute'],['Q3/report_q3.py'],['Q3/test_q3.py'],
 ['Q3/audit_q3.py']]
COMMANDS_Q4=[['Q4/solve_q4.py','--prepare'],['Q4/solve_q4.py','--precheck'],
 ['Q4/solve_q4.py','--compute-kind','42'],['Q4/solve_q4.py','--compute-kind','43'],
 ['Q4/solve_q4.py','--diagnostics-kind','42'],['Q4/solve_q4.py','--diagnostics-kind','43'],
 ['Q4/solve_q4.py','--deliver'],['Q4/test_q4.py'],['Q4/audit_q4.py']]

def commands(through):
 return COMMANDS_Q3 + (COMMANDS_Q4 if through=='Q4' else [])

def required_inputs(through):
 names=[f'附件/附件{i}.xlsx' for i in range(1,4)]
 names += [f'附件/附件5/result{i}.xlsx' for i in range(1,4)]
 if through=='Q4':
  names += ['附件/附件4.xlsx','附件/附件5/result4-2.xlsx','附件/附件5/result4-3.xlsx']
 return names

def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--through',choices=('Q3','Q4'),default='Q4' if (ROOT/'Q4').is_dir() else 'Q3',
                     help='run Q1 through the selected question (default: highest packaged question)')
 parser.add_argument('--check-inputs',action='store_true',help='check required file presence and exit without running solvers')
 options=parser.parse_args(argv)
 missing=[name for name in required_inputs(options.through) if not (ROOT/name).is_file()]
 missing_code=sorted({args[0] for args in commands(options.through) if not (ROOT/args[0]).is_file()})
 if missing or missing_code:
  if missing:
   print('Official inputs are not distributed in this public derivative. Obtain them lawfully and place these files under code/:',file=sys.stderr)
   for name in missing:print('  '+name,file=sys.stderr)
  if missing_code:
   print('This package does not contain the requested workflow:',file=sys.stderr)
   for name in missing_code:print('  '+name,file=sys.stderr)
  return 2
 if options.check_inputs:
  print('Required input and entry-point files are present. Workbook contents have not been validated.')
  return 0
 for args in commands(options.through):
  print('RUN',*args,flush=True)
  subprocess.run([sys.executable,*args],cwd=ROOT,check=True)
 print(f'{options.through} public workflow completed: solver, report, workbook, output tests and numerical audits passed. Original private freezes were not checked.')
 return 0

if __name__=='__main__':
 sys.exit(main())
