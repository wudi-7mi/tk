import asyncio

async def heavy_task():
  print("--")
  await asyncio.sleep(1)

async def main():
  async for _ in range(10):
    await heavy_task()

if __name__ == '__main__':
  asyncio.run(main())