; RUN: llc < %s -mtriple=mipsel -mcpu=mips32 -fast-isel -frame-pointer=all -relocation-model=pic < %s

; Test that negative array access don't crash constant synthesis when fast isel
; generates negative offsets.

define i16 @test() {
  %a = alloca [4 x i16], align 4
  %arrayidx = getelementptr inbounds [4 x i16], ptr %a, i32 0, i32 -2
  %b = load i16, ptr %arrayidx, align 2
  ret i16 %b
}

define void @test2() {
  %a = alloca [4 x i16], align 4
  %arrayidx = getelementptr inbounds [4 x i16], ptr %a, i32 0, i32 -2
  store i16 2, ptr %arrayidx, align 2
  ret void
}
