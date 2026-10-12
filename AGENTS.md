# AeroNode project instructions

- The user-confirmed GitHub repository is **https://github.com/abhishekkr10/AeroNode**. Publish this project's validated code and documentation updates only to that repository. Do not substitute a similarly spelled account or another repository.
- The user has requested continued implementation of the documented roadmap and publishing of validated updates. Verify the push destination and remote state before publishing. Preserve other work; never force-push, reset, clean unrelated files, or rewrite history.
- Core inference remains C++17 with ONNX Runtime. Preserve the existing Phase-4 exit policy, ZeroMQ snapshot architecture, and explicit-copy stage boundaries unless the current requested phase explicitly changes them.
- Generated ONNX models, checkpoints, datasets, benchmark binary inputs/raw records, local SDKs, credentials, and build output remain excluded from Git. Commit reproducible source and concise validation records.
- Numerical parity and execution-path tests are required for runtime changes. Use the newly built executables for regression checks. Record exact test counts, actual compiler output, measurement scope, and limitations.
- Performance claims require recorded measured evidence. Controlled RuntimeState values are not hardware sensor readings. Threshold fitting must separate calibration and evaluation samples.
