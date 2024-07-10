// eslint-disable-next-line
import { EventEmitter } from 'events';
import type { ILink } from '@comunica/bus-rdf-resolve-hypermedia-links';
import type { IQuerySource, IStatisticDereferencedLinks, Logger } from '@comunica/types';
import { from } from 'readable-stream';

export class StatisticLinkDereference implements IStatisticDereferencedLinks {
  public dereferenceOrder: ILink[];

  public statisticEvents: EventEmitter;

  public log: ((message: string, data?: (() => any)) => void);

  public constructor(log: (message: string, data?: (() => any)) => void) {
    this.dereferenceOrder = [];
    this.statisticEvents = new EventEmitter();

    this.log = log;
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

    // Log info if available
    this.log('Dereference Event', () => dereferencedLink);

    return true;
  }

  public addListener(cb: (arg0: ILink) => void): void {
    this.statisticEvents.addListener('data', cb);
  }

  public emit(data: ILink) {
    this.statisticEvents.emit('data', data);
  }
}
