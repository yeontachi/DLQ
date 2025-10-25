import json, os, socket, getpass, platform, time, inspect, hashlib, sys
from datetime import datetime
from typing import Dict, Any, Optional, Union
import pandas as pd

try:
    import psutil
except Exception:
    psutil = None

class ExcelLogger:
    """
    엑셀 자동 기록 유틸.
    - 워크북: results.xlsx (기본)
    - 시트: 실행한 스크립트 파일명(확장자 제외)
    - 각 로그는 한 행(row)으로 누적: Timestamp, Script, RunID, Tag, Params, Metrics...
    - 같은 스크립트에서 다양한 메트릭 키를 추가해도 자동으로 컬럼 확장
    """
    def __init__(self, excel_path: str = "results.xlsx", script_name: Optional[str] = None, tag: str = ""):
        self.excel_path = excel_path
        # 스크립트 이름 자동 추출
        if script_name is None:
            script_name = self._infer_script_name()
        self.sheet_name = self._sanitize_sheet_name(script_name)
        self.tag = tag
        # 실행 구분자
        self.run_id = self._make_run_id()

    def log(self,
            metrics: Dict[str, Union[int, float, str]],
            params: Optional[Dict[str, Any]] = None,
            extra: Optional[Dict[str, Any]] = None) -> None:
        """
        metrics: {"val_acc":0.91, "val_loss":0.23, "fps": 120, ...}
        params:  {"lr":3e-4, "batch_size":128, "epochs":50, ...}
        extra:   {"note":"first try", "seed":42, ...}
        """
        row = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Script": self.sheet_name,
            "RunID": self.run_id,
            "Tag": self.tag,
        }
        # 시스템/환경
        row.update(self._system_info())
        # 파라미터/추가정보(JSON 문자열)
        if params is not None:
            row["Params"] = json.dumps(params, ensure_ascii=False)
        if extra is not None:
            row["Extra"] = json.dumps(extra, ensure_ascii=False)
        # 메트릭
        for k, v in metrics.items():
            row[f"m_{k}"] = v

        # 엑셀 파일/시트 append
        self._append_row(self.excel_path, self.sheet_name, row)

    # ------------------ 내부 유틸 ------------------ #
    def _append_row(self, path: str, sheet: str, row_dict: Dict[str, Any]):
        df_new = pd.DataFrame([row_dict])

        if not os.path.exists(path):
            # 새 워크북 생성
            with pd.ExcelWriter(path, engine="openpyxl") as w:
                df_new.to_excel(w, sheet_name=sheet, index=False)
            return

        with pd.ExcelWriter(path, engine="openpyxl", mode="a", if_sheet_exists="overlay") as w:
            # 기존 시트 존재 여부 판단
            try:
                existing = pd.read_excel(path, sheet_name=sheet)
                # 컬럼 통일(새 키가 생기면 확장)
                all_cols = list(dict.fromkeys(list(existing.columns) + list(df_new.columns)))
                existing = existing.reindex(columns=all_cols)
                df_new = df_new.reindex(columns=all_cols)
                out = pd.concat([existing, df_new], ignore_index=True)
            except ValueError:
                # 시트 없음 → 새 시트 생성
                out = df_new
            out.to_excel(w, sheet_name=sheet, index=False)

    def _infer_script_name(self) -> str:
        # __main__ 파일 경로 또는 호출 스택으로 추정
        frame = inspect.stack()[-1]
        fname = frame.filename
        base = os.path.basename(fname)
        name, _ = os.path.splitext(base)
        return name

    def _sanitize_sheet_name(self, name: str) -> str:
        # 엑셀 시트명 제한 대응 (31자, 특수문자 불가)
        bad = [":", "\\", "/", "?", "*", "[", "]"]
        for b in bad:
            name = name.replace(b, "_")
        return name[:31]

    def _make_run_id(self) -> str:
        # 시간+PID 해시
        raw = f"{time.time()}_{os.getpid()}"
        return hashlib.sha1(raw.encode()).hexdigest()[:8]

    def _system_info(self) -> Dict[str, Any]:
        info = {
            "Host": socket.gethostname(),
            "User": getpass.getuser(),
            "OS": platform.platform(),
            "Python": sys.version.split()[0],
        }
        if psutil:
            try:
                info.update({
                    "CPU_Count": psutil.cpu_count(logical=True),
                    "RAM_GB": round(psutil.virtual_memory().total / (1024**3), 2),
                })
            except Exception:
                pass
        return info

# ---------- 데코레이터(선택): 함수/에폭 결과를 자동 기록 ---------- #
def log_returned_metrics_to_excel(excel_path: str = "results.xlsx", tag: str = ""):
    """
    함수가 dict(metrics) 를 반환하면 자동으로 엑셀에 append.
    - 시트명은 호출한 스크립트명
    - 사용 예시:
        @log_returned_metrics_to_excel(tag="sanity")
        def evaluate(...):
            return {"val_acc":0.9, "val_loss":0.3}
    """
    def deco(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if isinstance(result, dict):
                logger = ExcelLogger(excel_path=excel_path, tag=tag)
                logger.log(metrics=result)
            return result
        return wrapper
    return deco