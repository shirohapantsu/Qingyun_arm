# 原分支参考资料

来源：GitHub feature/vision，commit 4e8b5757ac9d634457adabb3863440f854319734。

- `astra_pro_calib.json`：原双目参数，平移以 mm 存储；只供原型诊断工具消费。
- `camera_intrinsics_legacy.json`：原 P3 格式转换产物，深度内参分辨率 1280×1024，D2C 平移仍是 mm；不能用于当前生产管线的 640×480/m 契约。
- `T_base_camera_identity.json`：恒等矩阵占位，不能作为手眼实测外参。
- 原始相机/标定快照：仅保留来源，不足以证明 CAL-040/041 通过。

新的运行标定模板在 configs/vision；现场产物经独立验收后由 vision.local.json 引用。
