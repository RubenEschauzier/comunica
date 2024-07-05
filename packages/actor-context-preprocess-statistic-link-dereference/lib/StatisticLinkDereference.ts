// eslint-disable-next-line
import { EventEmitter } from 'events';
import type { ILink } from '@comunica/bus-rdf-resolve-hypermedia-links';
import type { IQuerySource, IStatisticDereferencedLinks, Logger } from '@comunica/types';

export class StatisticLinkDereference implements IStatisticDereferencedLinks {
  public dereferenceOrder: ILink[];

  public statisticEvents: EventEmitter;

  protected logger: Logger | undefined;

  public constructor(logger?: Logger) {
    this.dereferenceOrder = [];

    this.statisticEvents = new EventEmitter();

    this.logger = logger;
  }

  public setDereferenced(link: ILink, source: IQuerySource) {
    const metadata: Record<string, any> = {
      type: source.constructor.name,
      dereferencedTimestamp: Date.now(),
      dereferenceOrder: this.dereferenceOrder.length,
      ...link.metadata,
    };

    const dereferencedLink: ILink = {
      url: link.url,
      metadata,
      context: link.context,
      transform: link.transform,
    };

    this.dereferenceOrder.push(dereferencedLink);

    this.statisticEvents.emit('data', dereferencedLink);

    // TODO: Update logging to be browser safe
    if (this.logger) {
      this.logger.trace('Dereference Event', {
        data: JSON.stringify({
          statistic: 'dereferencedLinks',
          dereferencedLinks: this.dereferenceOrder,
        }),
      });
    }

    return true;
  }

  public addListener(cb: (arg0: ILink) => void): void {
    this.statisticEvents.addListener('data', cb);
  }

  public emit(data: ILink) {
    this.statisticEvents.emit('data', data);
  }
}
