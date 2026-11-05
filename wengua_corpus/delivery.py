"""Authenticated encrypted delivery; private key never belongs in Git.

AES-256-CBC + HMAC-SHA256 encrypt-then-MAC. Independent keys are derived
from a random 256-bit master with labelled HMAC. Windows uses built-in .NET;
no new Python dependency and no edits to the frozen training runtime.
"""
from __future__ import annotations
import base64
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]
MAGIC=b'WQ4ENC1\x00'
LABEL=b'WenguaApp/course-v4/delivery-v1/'
MANIFEST=ROOT/'data/delivery/qwen_course_v4/manifest.json'

# No secret is placed in the command line or a PowerShell script file.
# This command reads only the already MAC-verified cipher/key/IV from stdin.
WINDOWS_DECRYPT=r"""
$ErrorActionPreference = 'Stop'
$r = [Console]::In.ReadToEnd() | ConvertFrom-Json
$a = [System.Security.Cryptography.Aes]::Create()
try {
  $a.Key = [Convert]::FromBase64String($r.key)
  $a.IV = [Convert]::FromBase64String($r.iv)
  $a.Mode = [System.Security.Cryptography.CipherMode]::CBC
  $a.Padding = [System.Security.Cryptography.PaddingMode]::PKCS7
  $c = [Convert]::FromBase64String($r.cipher)
  $d = $a.CreateDecryptor()
  try { $p = $d.TransformFinalBlock($c, 0, $c.Length) }
  finally { $d.Dispose() }
  [Console]::Out.Write([Convert]::ToBase64String($p))
} finally { $a.Dispose() }
"""


def derive_keys(master):
    if not isinstance(master,bytes) or len(master)!=32:raise ValueError('Delivery key must be 32 bytes')
    return tuple(hmac.new(master,LABEL+label,hashlib.sha256).digest() for label in (b'encryption',b'authentication'))


def authenticate(blob,master):
    if not isinstance(blob,bytes) or len(blob)<len(MAGIC)+16+16+32 or not blob.startswith(MAGIC):
        raise ValueError('Invalid encrypted delivery header')
    key,mac=derive_keys(master)
    if not hmac.compare_digest(hmac.new(mac,blob[:-32],hashlib.sha256).digest(),blob[-32:]):
        raise ValueError('Wrong delivery key or corrupted file; nothing was decrypted')
    iv=blob[len(MAGIC):len(MAGIC)+16];cipher=blob[len(MAGIC)+16:-32]
    if not cipher or len(cipher)%16:raise ValueError('Invalid ciphertext length')
    return key,iv,cipher


def windows_decrypt(key,iv,cipher,*,runner=subprocess.run):
    payload=json.dumps({k:base64.b64encode(v).decode('ascii') for k,v in
                        [('key',key),('iv',iv),('cipher',cipher)]})
    result=runner(['powershell.exe','-NoProfile','-NonInteractive','-Command',WINDOWS_DECRYPT],
                  input=payload,capture_output=True,text=True,timeout=120,check=False)
    if result.returncode:raise RuntimeError('Windows built-in AES decryption failed; no output file was installed')
    try:return base64.b64decode(result.stdout.strip(),validate=True)
    except ValueError as error:raise RuntimeError('Invalid built-in AES response') from error


def install(blob,master,output,*,expected_sha256,decrypt=windows_decrypt):
    """Never create/replace an output until authentication and plain SHA pass."""
    key,iv,cipher=authenticate(blob,master)
    plain=decrypt(key,iv,cipher)
    if hashlib.sha256(plain).hexdigest()!=expected_sha256:raise ValueError('Decrypted ZIP SHA256 mismatch')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        if hashlib.sha256(output.read_bytes()).hexdigest()!=expected_sha256:
            raise FileExistsError('Different existing file preserved: '+str(output))
        return output
    temp=None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent,prefix=output.name+'.',suffix='.pending',delete=False) as f:
            temp=Path(f.name);f.write(plain);f.flush();os.fsync(f.fileno())
        # The final path is never overwritten. Windows fails if another writer
        # creates it between the existence check and this rename.
        if output.exists():raise FileExistsError('Output appeared during installation; preserved')
        temp.rename(output);temp=None
    finally:
        if temp is not None:temp.unlink(missing_ok=True)
    return output


def main(argv=None):
    import argparse
    import getpass
    parser=argparse.ArgumentParser(description='Restore the exact reviewed ZIP from the encrypted Git delivery')
    parser.add_argument('--key',help='64 hex characters from this private conversation; never commit the key')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    manifest=json.loads(MANIFEST.read_text(encoding='utf8'))
    path=ROOT/manifest['file']
    blob=path.read_bytes()
    if hashlib.sha256(blob).hexdigest()!=manifest['ciphertext_sha256']:raise ValueError('Encrypted Git file changed')
    if os.name!='nt':raise RuntimeError('This entry uses Windows built-in AES; run it on the training computer')
    value=args.key or getpass.getpass('Paste the private delivery key: ')
    if not re.fullmatch(r'[0-9a-fA-F]{64}',value):raise ValueError('Delivery key must contain exactly 64 hex characters')
    output=args.output or Path.home()/'Downloads'/manifest['archive_name']
    path=install(blob,bytes.fromhex(value),output,expected_sha256=manifest['archive_sha256'])
    print('Dataset ZIP restored and SHA256 verified: '+str(path))
    print('Next in PowerShell: .\\run_qwen_course_v4_windows.cmd')
    return 0

if __name__=='__main__':raise SystemExit(main())
